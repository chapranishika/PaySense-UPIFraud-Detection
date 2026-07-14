package com.paysense.app.layer2

import android.content.Context
import android.util.Log
import com.paysense.app.layer1.ParsedTransaction

private const val TAG = "PaySense_Layer2"

// ──────────────────────────────────────────────────────────────────────────────
//  PayeeCacheRepository — Integration logic
//
//  NOTE: PaySenseDatabase is defined in PaySenseDatabase.kt (the authoritative
//  singleton with all three migrations). This file previously contained a
//  duplicate @Database class with version=1 and fallbackToDestructiveMigration().
//  That duplicate has been removed — it would have caused a version conflict
//  at runtime (two @Database classes targeting the same file, different versions).
//
//  The Repository now uses PaySenseDatabase.getInstance(context) from the
//  dedicated file, ensuring a single database singleton across the entire app.
//
//  The Repository owns the THREE-TIER resolution logic:
//    Tier 1 → Local Room cache lookup
//    Tier 2 → NLP classifier (stubbed here; implement with TFLite or API call)
//    Tier 3 → Human-in-the-Loop prompt (triggered via broadcast to UI)
//
//  This is the function you should walk the panel through — it is the
//  architectural heart of Layer 2.
// ──────────────────────────────────────────────────────────────────────────────
class PayeeCacheRepository private constructor(private val context: Context) {

    private val dao: PayeeDao =
        PaySenseDatabase.getInstance(context).payeeDao()

    companion object {
        @Volatile private var INSTANCE: PayeeCacheRepository? = null

        fun getInstance(context: Context): PayeeCacheRepository =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: PayeeCacheRepository(context.applicationContext).also { INSTANCE = it }
            }
    }

    // ── PUBLIC API ──────────────────────────────────────────────────────────

    /**
     * Tier-1 cache lookup. Returns the stored category string if this payee
     * has been seen before, or null on a cache miss.
     *
     * Called by SmsReceiver immediately after Gate 3 extraction.
     */
    suspend fun getCategoryForPayee(rawPayee: String): String? {
        val normalised = rawPayee.trim().lowercase()
        val cached     = dao.getCategoryForPayee(normalised)

        return if (cached != null) {
            Log.d(TAG, "💾  Tier 1 HIT | '$rawPayee' → '${cached.category}' (source=${cached.source})")
            cached.category
        } else {
            Log.d(TAG, "❓  Tier 1 MISS | '$rawPayee' not in local cache")
            null
        }
    }

    /**
     * Saves a user-confirmed category (Tier 3 HITL result) to the local cache.
     *
     * Called by the CategorySelectionBottomSheet when the user taps a category chip.
     * The `source = "user"` flag marks this as a high-confidence human label.
     */
    suspend fun saveUserCategory(rawPayee: String, category: String) {
        val normalised = rawPayee.trim().lowercase()
        val entry = PayeeCache(
            payeeName  = normalised,
            category   = category,
            source     = "user",
            confidence = null       // Human labels have no probabilistic confidence
        )
        dao.insertPayeeCategory(entry)
        Log.d(TAG, "✅  SAVED user category | '$normalised' → '$category'")
    }

    /**
     * Saves an NLP-inferred category (Tier 2 result) to the cache.
     *
     * Only called when NLP confidence is ≥ 0.65. Below that threshold,
     * the caller triggers the HITL UI prompt instead.
     *
     * @param confidence The classifier's softmax score (0.0 – 1.0).
     */
    suspend fun saveNlpCategory(rawPayee: String, category: String, confidence: Float) {
        val normalised = rawPayee.trim().lowercase()
        val entry = PayeeCache(
            payeeName  = normalised,
            category   = category,
            source     = "nlp",
            confidence = confidence
        )
        dao.insertPayeeCategory(entry)
        Log.d(TAG, "🤖  SAVED NLP category | '$normalised' → '$category' (conf=${"%.2f".format(confidence)})")
    }

    /**
     * ══════════════════════════════════════════════════════════════════════
     *  CORE INTEGRATION FUNCTION — Three-Tier Category Resolution
     *
     *  This is the function that demonstrates the full Layer 2 pipeline.
     *  Walk the panel through this function step by step.
     *
     *  @param txn The [ParsedTransaction] produced by Gate 3 in Layer 1.
     *  @return    A [CategoryResult] describing the resolution path taken.
     * ══════════════════════════════════════════════════════════════════════
     */
    suspend fun resolveCategory(txn: ParsedTransaction): CategoryResult {

        val payee = txn.payee

        // ── TIER 1: Local Room cache ─────────────────────────────────────────
        Log.d(TAG, "🔍  Tier 1 | Checking local cache for '${payee}'…")
        val cachedCategory = getCategoryForPayee(payee)

        if (cachedCategory != null) {
            // Fast path — no NLP call, no user prompt, no network request.
            return CategoryResult(
                category   = cachedCategory,
                source     = CategorySource.CACHE,
                confidence = null,
                requiresUserPrompt = false
            )
        }

        // ── TIER 2: NLP Classifier ───────────────────────────────────────────
        // Pass the raw SMS body (not just the payee name) for richer context.
        Log.d(TAG, "🤖  Tier 2 | Running NLP classifier on '${payee}'…")
        val nlpResult = runNlpClassifier(txn.rawBody, payee)

        if (nlpResult != null && nlpResult.second >= NLP_CONFIDENCE_THRESHOLD) {
            val (nlpCategory, nlpConfidence) = nlpResult

            Log.d(TAG, "✅  Tier 2 CONFIDENT | '$payee' → '$nlpCategory' (conf=${"%.2f".format(nlpConfidence)})")

            // Cache this for future transactions to this payee
            saveNlpCategory(payee, nlpCategory, nlpConfidence)

            return CategoryResult(
                category   = nlpCategory,
                source     = CategorySource.NLP,
                confidence = nlpConfidence,
                requiresUserPrompt = false
            )
        }

        // ── TIER 3: Human-in-the-Loop ────────────────────────────────────────
        // NLP confidence too low (or NLP returned null) — ask the user once.
        val lowConf = nlpResult?.second
        Log.d(TAG, "❓  Tier 3 HITL | NLP confidence too low (${"%.2f".format(lowConf ?: 0f)}) → user prompt required for '${payee}'")

        return CategoryResult(
            category           = "Uncategorized",  // Temporary — replaced when user responds
            source             = CategorySource.PENDING_USER,
            confidence         = lowConf,
            requiresUserPrompt = true              // SmsReceiver broadcasts SHOW_CATEGORY_PROMPT
        )
    }

    // ── PRIVATE HELPERS ─────────────────────────────────────────────────────

    /**
     * Stub for the NLP classifier call.
     *
     * In production, replace this with one of:
     *   Option A: TFLite on-device inference using your FinText-6K model
     *             (zero latency, works offline, ~15MB model file)
     *   Option B: HTTP call to a /classify endpoint on your FastAPI server
     *             (always up-to-date model, requires internet)
     *
     * @return Pair<category, confidence> or null if classification fails entirely.
     */
    private suspend fun runNlpClassifier(rawBody: String, payee: String): Pair<String, Float>? {
        // ── Keyword shortcut rules (high-confidence, zero-latency) ────────────
        // Common merchant names that are globally unambiguous. This replaces
        // an NLP call for the most frequent payees and reduces API usage.
        val keywordRules = mapOf(
            "zomato"     to ("Food"     to 0.99f),
            "swiggy"     to ("Food"     to 0.99f),
            "dominos"    to ("Food"     to 0.99f),
            "irctc"      to ("Travel"   to 0.99f),
            "ola"        to ("Travel"   to 0.97f),
            "uber"       to ("Travel"   to 0.97f),
            "amazon"     to ("Shopping" to 0.98f),
            "flipkart"   to ("Shopping" to 0.98f),
            "netflix"    to ("Entertainment" to 0.98f),
            "hotstar"    to ("Entertainment" to 0.98f),
            "electricity" to ("Utilities" to 0.96f),
            "bsnl"       to ("Recharge" to 0.95f),
            "airtel"     to ("Recharge" to 0.95f),
            "jio"        to ("Recharge" to 0.95f),
        )

        val payeeLower = payee.lowercase()
        for ((keyword, result) in keywordRules) {
            if (payeeLower.contains(keyword)) {
                Log.d(TAG, "⚡  Keyword rule matched: '$payee' contains '$keyword'")
                return result
            }
        }

        // ── TODO: Replace with actual TFLite / API NLP call here ─────────────
        // val response = nlpApiService.classify(rawBody)
        // return Pair(response.category, response.confidence)

        // Returning null signals to the caller that NLP is not available
        // and Tier 3 HITL must be triggered.
        Log.d(TAG, "⚙️  NLP stub: no keyword rule matched for '${payee}' → falling to HITL")
        return null
    }

    companion object {
        // Minimum NLP confidence to auto-assign a category without prompting.
        // Below this threshold, the HITL prompt is shown to the user.
        const val NLP_CONFIDENCE_THRESHOLD = 0.65f
    }
}

// ──────────────────────────────────────────────────────────────────────────────
//  Supporting data types
// ──────────────────────────────────────────────────────────────────────────────

/** The output of [PayeeCacheRepository.resolveCategory]. */
data class CategoryResult(
    val category           : String,            // The resolved or temporary category
    val source             : CategorySource,    // How it was resolved
    val confidence         : Float?,            // NLP confidence (null if user/cache)
    val requiresUserPrompt : Boolean            // True → show HITL bottom sheet
)

enum class CategorySource {
    CACHE,          // Answered from local Room DB — fastest path
    NLP,            // Answered by classifier with confidence ≥ threshold
    PENDING_USER    // Awaiting user input in the HITL bottom sheet
}
