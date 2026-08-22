package com.paysense.app.layer2

import android.content.Context
import android.util.Log
import com.paysense.app.layer1.ParsedTransaction
import com.paysense.app.layer3.FraudApiService

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

        // Minimum NLP confidence to auto-assign a category without prompting.
        // Below this threshold, the HITL prompt is shown to the user.
        const val NLP_CONFIDENCE_THRESHOLD = 0.65f
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
     * NLP classifier call — Option B (HTTP call to /classify on the FastAPI
     * server), chosen over on-device TFLite because it lets the server-side
     * model (retrained on fresh data) improve without an app update, and
     * because PaySense already has a live backend + Retrofit client for
     * /predict that this reuses.
     *
     * Order of operations:
     *   1. Keyword shortcut rules (NlpKeywordRules) — high-confidence,
     *      zero-latency, zero-network-cost for the most common merchants.
     *   2. POST /classify with the raw SMS body — the FinText-6K-trained
     *      TF-IDF + LinearSVC classifier on the backend. Only ever returns
     *      one of five classes: Food, Travel, EMI, Investment, Shopping.
     *      That's a real, honest limitation versus the app's full HITL
     *      category set (Bills, Grocery, Entertainment, Healthcare, Misc) —
     *      this is not silently mapped or faked; a category outside those
     *      five simply won't come from this tier and falls through to
     *      Tier 3 where the user can pick from the full set (which now
     *      also includes EMI and Investment as chips).
     *
     * @return Pair<category, confidence> or null if classification fails
     *         entirely (network error, no keyword match and server
     *         unavailable) — the caller then falls to Tier 3 HITL.
     */
    private suspend fun runNlpClassifier(rawBody: String, payee: String): Pair<String, Float>? {
        // ── Keyword shortcut rules (high-confidence, zero-latency) ────────────
        // Common merchant names that are globally unambiguous. This replaces
        // an NLP call for the most frequent payees and reduces API usage.
        // The rule table + matching logic live in NlpKeywordRules (a pure,
        // Context/Room-free object) so they can be unit tested directly.
        val match = NlpKeywordRules.classify(payee)
        if (match != null) {
            Log.d(TAG, "⚡  Keyword rule matched: '$payee'")
            return match
        }

        // ── Tier 2 proper: POST /classify with the raw SMS narration ─────────
        // FraudApiService owns the Retrofit client/PaySenseApi instance
        // (same pattern as predictFraud() / getWeeklyInsights()), so we
        // reuse its singleton instead of building a second Retrofit client.
        val nlpResult = FraudApiService.getInstance(context).classifyCategory(rawBody)
        if (nlpResult != null) {
            Log.d(TAG, "🧠  NLP API responded for '$payee' → '${nlpResult.first}' (conf=${nlpResult.second})")
            return nlpResult
        }

        // Returning null signals to the caller that NLP is not available
        // (keyword miss AND API call failed/returned nothing) and Tier 3
        // HITL must be triggered.
        Log.d(TAG, "⚙️  NLP unavailable: no keyword rule and no API result for '${payee}' → falling to HITL")
        return null
    }

}

// ──────────────────────────────────────────────────────────────────────────────
//  NlpKeywordRules — pure keyword-matching logic extracted out of
//  PayeeCacheRepository so it can be unit tested without needing a Context
//  or a Room database. Behavior is identical to the inline table this
//  replaced; only the visibility/location changed.
// ──────────────────────────────────────────────────────────────────────────────
internal object NlpKeywordRules {

    val rules: Map<String, Pair<String, Float>> = mapOf(
        "zomato"      to ("Food"          to 0.99f),
        "swiggy"      to ("Food"          to 0.99f),
        "dominos"     to ("Food"          to 0.99f),
        "irctc"       to ("Travel"        to 0.99f),
        "ola"         to ("Travel"        to 0.97f),
        "uber"        to ("Travel"        to 0.97f),
        "amazon"      to ("Shopping"      to 0.98f),
        "flipkart"    to ("Shopping"      to 0.98f),
        "netflix"     to ("Entertainment" to 0.98f),
        "hotstar"     to ("Entertainment" to 0.98f),
        "electricity" to ("Utilities"     to 0.96f),
        "bsnl"        to ("Recharge"      to 0.95f),
        "airtel"      to ("Recharge"      to 0.95f),
        "jio"         to ("Recharge"      to 0.95f),
    )

    /**
     * Returns (category, confidence) for the first keyword the payee name
     * contains (case-insensitive), or null if no keyword rule matches —
     * the caller should then fall back to HITL.
     */
    fun classify(payee: String): Pair<String, Float>? {
        val payeeLower = payee.lowercase()
        for ((keyword, result) in rules) {
            if (payeeLower.contains(keyword)) {
                return result
            }
        }
        return null
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
