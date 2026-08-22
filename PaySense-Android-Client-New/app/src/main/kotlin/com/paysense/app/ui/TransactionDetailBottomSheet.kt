package com.paysense.app.ui

import android.content.res.ColorStateList
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.paysense.app.R
import com.paysense.app.databinding.ItemRiskFactorRowBinding
import com.paysense.app.databinding.LayoutBottomSheetTransactionDetailBinding
import com.paysense.app.layer2.PaySenseDatabase
import com.paysense.app.layer2.TransactionHistory
import kotlinx.coroutines.launch
import java.text.NumberFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

/**
 * ============================================================================
 *  TransactionDetailBottomSheet.kt
 *
 *  Combines wireframe screens 3 ("Transaction Analysis") and 4
 *  ("Risk Details (AI)") into a single sheet opened by tapping a row in the
 *  Recent Transactions list.
 *
 *  EVERY value shown here comes from the real, already-persisted
 *  TransactionHistory row (schema v6 — see PaySenseDatabase.MIGRATION_5_6
 *  and FraudApiService.scoreTransaction()), via RiskFactorAnalyzer:
 *    - Gauge number/fill/colour  <- fraudScore, alertLevel
 *    - Risk-breakdown rows       <- amountDeviationScore, isNightTransaction,
 *                                    newDeviceFlag, ipLocationMismatch
 *    - AI Suggestion text        <- a deterministic if-chain over which of
 *                                    those factors are actually elevated
 *
 *  Transactions inserted before this migration have null factor fields;
 *  those rows render "Not available for this transaction" (RiskLevel.UNKNOWN)
 *  rather than a fabricated Low/zero value.
 * ============================================================================
 */
class TransactionDetailBottomSheet : BottomSheetDialogFragment() {

    companion object {
        const val TAG = "TransactionDetailBottomSheet"
        private const val ARG_TXN_ID = "txn_id"

        fun newInstance(txnId: String): TransactionDetailBottomSheet =
            TransactionDetailBottomSheet().apply {
                arguments = Bundle().apply { putString(ARG_TXN_ID, txnId) }
            }
    }

    private var _binding: LayoutBottomSheetTransactionDetailBinding? = null
    private val binding get() = _binding!!

    private val txnId by lazy { arguments?.getString(ARG_TXN_ID) ?: "" }
    private val currencyFmt = NumberFormat.getCurrencyInstance(Locale("en", "IN")).apply {
        maximumFractionDigits = 0
    }
    private val timestampFmt = SimpleDateFormat("dd MMM yyyy, hh:mm a", Locale.getDefault())

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = LayoutBottomSheetTransactionDetailBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        (dialog as? BottomSheetDialog)?.behavior?.apply {
            state = BottomSheetBehavior.STATE_EXPANDED
            skipCollapsed = true
        }

        binding.btnDetailClose.setOnClickListener { dismiss() }

        viewLifecycleOwner.lifecycleScope.launch {
            val txn = PaySenseDatabase.getInstance(requireContext())
                .transactionDao()
                .getTransactionById(txnId)
            if (txn == null) {
                dismiss()
            } else {
                render(txn)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun render(txn: TransactionHistory) {
        val ctx = requireContext()

        binding.tvDetailPayee.text = txn.payee.ifBlank { "Unknown Payee" }
        binding.tvDetailAmount.text = currencyFmt.format(txn.amount)
        binding.tvDetailTimestamp.text = if (txn.timestamp > 0L)
            timestampFmt.format(Date(txn.timestamp)) else txn.date

        // ── Gauge + risk pill + summary — driven by real fraudScore/alertLevel ──
        val scoreOutOf100 = (txn.fraudScore.coerceIn(0.0, 1.0) * 100).roundToInt()
        binding.tvDetailScore.text = "$scoreOutOf100/100"

        val (bgRes, textRes, pillLabel, summary) = when (txn.alertLevel) {
            "high"   -> AlertVisuals(R.color.colorPillDangerBg, R.color.colorAlertHigh,
                "HIGH RISK", "This transaction is significantly unusual — please review carefully.")
            "medium" -> AlertVisuals(R.color.colorPillReviewBg, R.color.colorAlertMedium,
                "MEDIUM RISK", "This transaction is slightly unusual.")
            "low"    -> AlertVisuals(R.color.colorPillSafeBg, R.color.colorAlertLow,
                "LOW RISK", "This transaction shows minor deviation from your usual pattern.")
            else     -> AlertVisuals(R.color.colorPillSafeBg, R.color.colorAlertNone,
                "SAFE TRANSACTION", "Your spending behavior is normal. No unusual patterns detected.")
        }
        val severityColor = ContextCompat.getColor(ctx, textRes)

        binding.gaugeDetailRisk.setProgressColor(severityColor)
        binding.gaugeDetailRisk.progress = txn.fraudScore.coerceIn(0.0, 1.0).toFloat()

        binding.tvDetailRiskPill.text = pillLabel
        binding.tvDetailRiskPill.setTextColor(severityColor)
        binding.tvDetailRiskPill.backgroundTintList =
            ColorStateList.valueOf(ContextCompat.getColor(ctx, bgRes))
        binding.tvDetailRiskSummary.text = summary

        // ── Risk-breakdown rows — real per-factor values, fixed display order ──
        val rows = RiskFactorAnalyzer.rows(txn)
        val rowBindings = listOf(binding.rowAmount, binding.rowTime, binding.rowDevice, binding.rowLocation)
        rowBindings.zip(rows).forEach { (rowBinding, row) ->
            bindFactorRow(rowBinding, row)
        }

        // ── AI Suggestion — deterministic logic over the real elevated factors ──
        binding.tvDetailSuggestion.text = RiskFactorAnalyzer.suggestion(txn)
    }

    private data class AlertVisuals(val bgRes: Int, val textRes: Int, val label: String, val summary: String)

    private fun bindFactorRow(row: ItemRiskFactorRowBinding, factor: RiskFactorRow) {
        val ctx = requireContext()
        row.tvFactorLabel.text = factor.label
        row.tvFactorDetail.text = factor.detail

        val (bgRes, textRes, pillText) = when (factor.level) {
            RiskLevel.LOW     -> Triple(R.color.colorPillSafeBg, R.color.colorAlertNone, "Low")
            RiskLevel.MEDIUM  -> Triple(R.color.colorPillReviewBg, R.color.colorAlertMedium, "Medium")
            RiskLevel.HIGH    -> Triple(R.color.colorPillDangerBg, R.color.colorAlertHigh, "High")
            RiskLevel.UNKNOWN -> Triple(R.color.colorStatPillBg, R.color.colorMuted, "N/A")
        }
        val color = ContextCompat.getColor(ctx, textRes)

        row.tvFactorPill.text = pillText
        row.tvFactorPill.setTextColor(color)
        row.tvFactorPill.backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(ctx, bgRes))
        row.barFactorFill.backgroundTintList = ColorStateList.valueOf(color)

        val fraction = RiskFactorAnalyzer.barFraction(factor).coerceIn(0f, 1f)
        (row.barFactorFill.layoutParams as LinearLayout.LayoutParams).weight = fraction
        (row.barFactorSpacer.layoutParams as LinearLayout.LayoutParams).weight = 1f - fraction
        row.barFactorFill.requestLayout()
        row.barFactorSpacer.requestLayout()
    }
}
