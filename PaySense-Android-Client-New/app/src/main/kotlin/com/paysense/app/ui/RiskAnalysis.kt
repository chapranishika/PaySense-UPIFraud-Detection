package com.paysense.app.ui

import com.paysense.app.layer2.TransactionHistory
import kotlin.math.abs

/**
 * ============================================================================
 *  RiskAnalysis.kt
 *
 *  Derives the "Risk Breakdown" / "Why is this risky?" rows and the AI
 *  suggestion text shown on TransactionDetailBottomSheet from the REAL
 *  per-transaction signals persisted on TransactionHistory (schema v6:
 *  amountDeviationScore, isNightTransaction, newDeviceFlag,
 *  ipLocationMismatch — see PaySenseDatabase.MIGRATION_5_6). Nothing here is
 *  fabricated: every label and every sentence is a deterministic function of
 *  a real stored value, and rows for pre-migration transactions (null
 *  fields) report UNKNOWN rather than a fake Low/zero.
 * ============================================================================
 */
enum class RiskLevel { LOW, MEDIUM, HIGH, UNKNOWN }

data class RiskFactorRow(
    val label: String,
    val level: RiskLevel,
    val detail: String
)

object RiskFactorAnalyzer {

    /**
     * Amount-deviation z-score → Low/Medium/High.
     * Documented thresholds (not model-derived — a reasonable, explicit
     * choice for presentation purposes only; the model itself sees the raw
     * z-score, not this bucketing):
     *   |z| < 1        -> Low     (within one standard deviation of usual spend)
     *   1 <= |z| <= 3   -> Medium  (noticeably off the user's usual pattern)
     *   |z| > 3        -> High    (statistically extreme for this user)
     */
    fun amountLevel(z: Double?): RiskLevel = when {
        z == null -> RiskLevel.UNKNOWN
        abs(z) > 3.0 -> RiskLevel.HIGH
        abs(z) >= 1.0 -> RiskLevel.MEDIUM
        else -> RiskLevel.LOW
    }

    /**
     * Boolean signals (isNightTransaction / newDeviceFlag / ipLocationMismatch)
     * have no magnitude — they are either flagged or not. We deliberately map
     * them straight to HIGH/LOW rather than inventing a fake Medium midpoint
     * or a fabricated bar-length gradient for a binary signal.
     */
    fun boolLevel(flag: Boolean?): RiskLevel = when (flag) {
        null -> RiskLevel.UNKNOWN
        true -> RiskLevel.HIGH
        false -> RiskLevel.LOW
    }

    /** Risk-breakdown rows for the "Transaction Analysis" card. */
    fun rows(txn: TransactionHistory): List<RiskFactorRow> = listOf(
        RiskFactorRow(
            label = "Unusual Amount",
            level = amountLevel(txn.amountDeviationScore),
            detail = txn.amountDeviationScore?.let {
                "%.2f".format(abs(it)) + "σ from your usual spend"
            } ?: "Not available for this transaction"
        ),
        RiskFactorRow(
            label = "Unusual Time",
            level = boolLevel(txn.isNightTransaction),
            detail = when (txn.isNightTransaction) {
                true  -> "Made between 10 PM and 6 AM"
                false -> "Within your usual active hours"
                null  -> "Not available for this transaction"
            }
        ),
        RiskFactorRow(
            label = "Device Change",
            level = boolLevel(txn.newDeviceFlag),
            detail = when (txn.newDeviceFlag) {
                true  -> "Sent from an unrecognised device"
                false -> "Recognised device"
                null  -> "Not available for this transaction"
            }
        ),
        RiskFactorRow(
            label = "Location Change",
            level = boolLevel(txn.ipLocationMismatch),
            detail = when (txn.ipLocationMismatch) {
                true  -> "IP-derived location doesn't match your usual location"
                false -> "Matches your usual location"
                null  -> "Not available for this transaction"
            }
        )
    )

    /** True when at least one real factor value is present on this row (v6+ transaction). */
    fun hasAnyData(txn: TransactionHistory): Boolean =
        txn.amountDeviationScore != null || txn.isNightTransaction != null ||
        txn.newDeviceFlag != null || txn.ipLocationMismatch != null

    /** 0f..1f bar-fill fraction for the "Why is this risky?" bars. UNKNOWN renders an empty bar. */
    fun barFraction(row: RiskFactorRow): Float = when (row.level) {
        RiskLevel.LOW -> 0.22f
        RiskLevel.MEDIUM -> 0.6f
        RiskLevel.HIGH -> 0.95f
        RiskLevel.UNKNOWN -> 0f
    }

    /**
     * Conditional AI-suggestion text — a small deterministic if-chain over
     * which real factors are actually elevated for this transaction. No LLM
     * call, no canned copy unrelated to the transaction's own data.
     */
    fun suggestion(txn: TransactionHistory): String {
        if (!hasAnyData(txn)) {
            return "Risk-factor details aren't available for this transaction — it was recorded " +
                   "before this feature was added."
        }

        val reasons = mutableListOf<String>()
        if (txn.newDeviceFlag == true) reasons += "verifying the device this was sent from"
        if (txn.ipLocationMismatch == true) reasons += "confirming this matches your current location"
        if (txn.isNightTransaction == true) reasons += "double-checking a late-night transfer like this"
        when (amountLevel(txn.amountDeviationScore)) {
            RiskLevel.HIGH -> reasons += "reviewing this amount carefully — it's far outside your usual range"
            RiskLevel.MEDIUM -> reasons += "reviewing this amount, which is higher than usual for you"
            else -> {}
        }

        return if (reasons.isEmpty())
            "Your spending behavior is normal. No unusual patterns detected."
        else
            "Consider " + reasons.joinToString(", ") + "."
    }
}
