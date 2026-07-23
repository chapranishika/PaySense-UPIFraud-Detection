package com.paysense.app.layer1

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.util.Log
import com.paysense.app.layer2.PayeeCacheRepository
import com.paysense.app.layer3.FraudApiService
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout

/**
 * ============================================================================
 *  SmsReceiver.kt  (v2 — goAsync() fix)
 *
 *  BUG FIXED (v1 → v2):
 *
 *  v1 (buggy):
 *    override fun onReceive(context: Context, intent: Intent) {
 *        val scope = CoroutineScope(Dispatchers.IO)   // NEW scope per SMS
 *        scope.launch { /* Room + Retrofit */ }
 *    }
 *    Problem 1 — ORPHANED COROUTINES:
 *      onReceive() returns immediately after launching the coroutine.
 *      Android's BroadcastReceiver lifecycle then ends. If the process
 *      is under memory pressure, Android may kill it before the coroutine
 *      finishes, silently dropping fraud scores and Room writes.
 *    Problem 2 — UNCONTROLLED SCOPE:
 *      The CoroutineScope has no parent, no cancellation hook, and no
 *      timeout. Under a burst of SMS messages, dozens of independent
 *      scopes pile up simultaneously all hitting Room and the network.
 *
 *  v2 (fixed) — goAsync() pattern:
 *
 *    goAsync() is the Android-blessed mechanism for doing async work in
 *    a BroadcastReceiver. It:
 *      1. Returns a PendingResult token that tells Android "I'm not done yet."
 *      2. Keeps the process alive past the end of onReceive().
 *      3. finish() on the PendingResult signals to Android that the work
 *         is complete and the process may now be eligible for reclaim.
 *
 *    We combine goAsync() with GlobalScope.launch + withTimeout:
 *      - GlobalScope is appropriate here (not a leak) because:
 *          a) The PendingResult.finish() call in the finally block ensures
 *             Android releases the wake-lock regardless of success/failure.
 *          b) withTimeout(RECEIVER_TIMEOUT_MS) bounds the maximum lifetime.
 *          c) BroadcastReceivers have no lifecycle object to scope against.
 *        This is the standard pattern recommended in Android documentation
 *        for async BroadcastReceiver work.
 *      - @OptIn(DelicateCoroutinesApi::class) is required because Kotlin
 *        warns that GlobalScope should be used carefully. The warning is
 *        correct in general; this is one of the few legitimate use cases.
 *
 *  TIMEOUT RATIONALE:
 *    Android allows BroadcastReceivers ~10 seconds before ANR.
 *    We set RECEIVER_TIMEOUT_MS = 9_000 (9 seconds) to stay within
 *    the ANR window and ensure finish() is always called before Android
 *    forcibly reclaims the receiver. Room operations take ~50ms;
 *    Retrofit calls take ~200ms on a fast WiFi. 9 seconds is generous.
 * ============================================================================
 */
class SmsReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "PaySense_Layer1"
        private const val RECEIVER_TIMEOUT_MS = 9_000L  // 9s — within Android's ANR window

        // ── Gate 1: TRAI Sender ID regex ─────────────────────────────────────
        private val GATE1_SENDER_REGEX = Regex(
            pattern = "^[A-Z]{2}-[A-Z0-9]{4,6}$",
            options  = setOf(RegexOption.IGNORE_CASE)
        )

        // ── Gate 2: Transaction keyword regex ────────────────────────────────
        private val GATE2_KEYWORD_REGEX = Regex(
            pattern = """(?i)\b(debited|credited|upi|rs\.|inr|transaction|payment)\b"""
        )

        // ── Gate 3: Named-group extraction regex ──────────────────────────────
        private val GATE3_EXTRACTION_REGEX = Regex(
            pattern =
                """(?i)(?:Rs\.?|INR|₹)\s*(?<amount>[\d,]+(?:\.\d{1,2})?)""" +
                """|(?:to|at|towards)\s+(?<payee>[A-Za-z0-9@._\-\s]{2,40}?)(?=\s+(?:for|on|via|ref|\d))""" +
                """|(?:UPI\s*Ref\.?|Ref\s*No\.?|Txn\s*[Ii][Dd])[:\s]*(?<txnId>[A-Z0-9]{8,20})""" +
                """|(?<date>\d{1,2}[-/]\w{3}[-/]\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})""",
            options = setOf(RegexOption.MULTILINE)
        )
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  onReceive — the BroadcastReceiver entry point
    //
    //  This method MUST return quickly (Android requires it in < 10s for ANR).
    //  All I/O work (Room + Retrofit) is dispatched via goAsync() so
    //  onReceive() returns in microseconds while the work continues safely.
    // ──────────────────────────────────────────────────────────────────────────
    @OptIn(DelicateCoroutinesApi::class)
    override fun onReceive(context: Context, intent: Intent) {

        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) return

        val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent)
        if (messages.isNullOrEmpty()) return

        // goAsync() MUST be called synchronously inside onReceive() before
        // returning. It returns a PendingResult that keeps the process alive.
        val pendingResult = goAsync()

        GlobalScope.launch(Dispatchers.IO) {
            try {
                // withTimeout cancels the coroutine if it takes too long,
                // ensuring finish() is always called via the finally block.
                withTimeout(RECEIVER_TIMEOUT_MS) {
                    for (message in messages) {
                        val sender    = message.originatingAddress ?: continue
                        val body      = message.messageBody        ?: continue
                        val timestamp = message.timestampMillis

                        Log.d(TAG, "━━━ SMS received | From: $sender")

                        // Run the three gates
                        if (!passesGate1(sender)) continue
                        if (!passesGate2(body))   continue
                        val parsed = applyGate3(sender, body, timestamp) ?: continue

                        Log.d(TAG, "✅  All gates passed | $parsed")
                        dispatchParsedTransaction(context, parsed)
                    }
                }
            } catch (e: Exception) {
                // Catches both TimeoutCancellationException and any unexpected error.
                // Log and swallow — we must not let an exception prevent finish().
                Log.e(TAG, "❌  Receiver error: ${e.message}")
            } finally {
                // CRITICAL: finish() MUST always be called.
                // If finish() is never called, the PendingResult leaks the
                // process wake-lock and Android cannot reclaim the receiver slot.
                pendingResult.finish()
                Log.d(TAG, "🔓  PendingResult.finish() called — receiver released")
            }
        }
        // onReceive() returns here immediately (~microseconds after being called).
        // The GlobalScope coroutine continues running in the background.
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  GATE 1 — TRAI Sender ID format check
    // ──────────────────────────────────────────────────────────────────────────
    private fun passesGate1(sender: String): Boolean {
        val passes = GATE1_SENDER_REGEX.matches(sender.trim())
        return if (passes) {
            Log.d(TAG, "🟢  Gate 1 PASS | $sender")
            true
        } else {
            Log.d(TAG, "🔴  Gate 1 FAIL | Not TRAI format: $sender")
            false
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  GATE 2 — Transaction keyword check
    // ──────────────────────────────────────────────────────────────────────────
    private fun passesGate2(body: String): Boolean {
        val match = GATE2_KEYWORD_REGEX.containsMatchIn(body)
        return if (match) {
            val kw = GATE2_KEYWORD_REGEX.find(body)?.value
            Log.d(TAG, "🟢  Gate 2 PASS | keyword='$kw'")
            true
        } else {
            Log.d(TAG, "🔴  Gate 2 FAIL | No transaction keyword")
            false
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  GATE 3 — Named-group regex extraction
    //  Returns null if the mandatory 'amount' field cannot be extracted.
    //  Null → quarantine log (not silent discard).
    // ──────────────────────────────────────────────────────────────────────────
    private fun applyGate3(sender: String, body: String, timestamp: Long): ParsedTransaction? {
        var amount: String? = null
        var payee : String? = null
        var txnId : String? = null
        var date  : String? = null

        for (result in GATE3_EXTRACTION_REGEX.findAll(body)) {
            if (amount == null) amount = result.groups["amount"]?.value?.replace(",", "")
            if (payee  == null) payee  = result.groups["payee"]?.value?.trim()
            if (txnId  == null) txnId  = result.groups["txnId"]?.value?.trim()
            if (date   == null) date   = result.groups["date"]?.value?.trim()
        }

        Log.d(TAG, "🔵  Gate 3 extract | amount=$amount payee=$payee txnId=$txnId")

        if (amount == null) {
            Log.w(TAG, "⚠️  Gate 3 QUARANTINE | amount missing — non-standard template from $sender")
            // TODO: write to QuarantineLog Room entity for manual user review
            return null
        }

        return ParsedTransaction(
            senderId  = sender,
            rawBody   = body,
            amount    = amount.toDoubleOrNull() ?: return null,
            payee     = payee  ?: "Unknown Payee",
            txnId     = txnId  ?: "TXN_${timestamp}",
            date      = date   ?: "",
            timestamp = timestamp
        )
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  DISPATCH — Layer 2 cache check then Layer 3 fraud scoring
    //
    //  Called from within the goAsync() coroutine — already on Dispatchers.IO.
    //  No additional coroutine launching needed here.
    // ──────────────────────────────────────────────────────────────────────────
    private suspend fun dispatchParsedTransaction(context: Context, txn: ParsedTransaction) {
        val repository = PayeeCacheRepository.getInstance(context)
        val cachedCategory = repository.getCategoryForPayee(txn.payee)

        if (cachedCategory != null) {
            Log.d(TAG, "💾  Cache HIT | ${txn.payee} → $cachedCategory")
            FraudApiService.getInstance(context).scoreTransaction(txn, cachedCategory)
        } else {
            Log.d(TAG, "❓  Cache MISS | ${txn.payee} — triggering HITL prompt")

            // Score with "Uncategorized" immediately so fraud checking is not
            // blocked waiting for the user to pick a category.
            FraudApiService.getInstance(context).scoreTransaction(txn, "Uncategorized")

            // Broadcast to MainActivity to show the CategoryBottomSheet.
            // RECEIVER_NOT_EXPORTED is handled by MainActivity's registerReceiver.
            val uiIntent = Intent("com.paysense.SHOW_CATEGORY_PROMPT").apply {
                putExtra("payee_name", txn.payee)
                putExtra("txn_id",    txn.txnId)
                putExtra("amount",    txn.amount)
                setPackage(context.packageName)  // restrict to our own app
            }
            context.sendBroadcast(uiIntent)
        }
    }
}

// ── Output contract from Gate 3 — passed between all three layers ─────────────
data class ParsedTransaction(
    val senderId  : String,
    val rawBody   : String,
    val amount    : Double,
    val payee     : String,
    val txnId     : String,
    val date      : String,
    val timestamp : Long
)
