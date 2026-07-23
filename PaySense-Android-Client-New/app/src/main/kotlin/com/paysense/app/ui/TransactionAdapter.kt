package com.paysense.app.ui

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.paysense.app.R
import com.paysense.app.databinding.ItemTransactionBinding
import com.paysense.app.layer2.TransactionHistory
import java.text.NumberFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * ============================================================================
 *  TransactionAdapter.kt
 *  RecyclerView adapter for the dashboard transaction list.
 *
 *  Uses ListAdapter (backed by DiffUtil) for efficient, animated updates
 *  when new transactions arrive from Room — only changed rows are redrawn.
 *
 *  Key responsibilities:
 *    • Format amount as Indian rupee string  (₹1,24,500.00)
 *    • Pick a category icon from the icon map
 *    • Apply fraud tinting when isFraud=true  (red card, warning icon)
 *    • Show/hide the fraud score sub-label
 * ============================================================================
 */
class TransactionAdapter(
    private val onItemClick: (TransactionHistory) -> Unit = {}
) : ListAdapter<TransactionHistory, TransactionAdapter.TxnViewHolder>(DiffCallback) {

    // ── DiffUtil — compares old and new lists so only changed rows animate ────
    private object DiffCallback : DiffUtil.ItemCallback<TransactionHistory>() {
        override fun areItemsTheSame(old: TransactionHistory, new: TransactionHistory) =
            old.txnId == new.txnId                // same DB row = same transaction ID

        override fun areContentsTheSame(old: TransactionHistory, new: TransactionHistory) =
            old == new                            // data class equality covers all fields
    }

    // ── Currency formatter using Indian locale (₹ with lakh separators) ──────
    private val currencyFormat = NumberFormat.getCurrencyInstance(
        Locale("en", "IN")
    ).apply { maximumFractionDigits = 2 }

    // ── Date formatter — "26 Apr", "01 Jan" ──────────────────────────────────
    private val dateFormat = SimpleDateFormat("dd MMM", Locale.getDefault())

    // ── Category icon map — key matches Layer 2 category strings ─────────────
    private val categoryIconMap = mapOf(
        "Food"          to R.drawable.ic_receipt,
        "Food & Dining" to R.drawable.ic_receipt,
        "Travel"        to R.drawable.ic_payment,
        "Shopping"      to R.drawable.ic_payment,
        "Bills"         to R.drawable.ic_receipt,
        "Utilities"     to R.drawable.ic_payment,
        "Recharge"      to R.drawable.ic_payment,
        "Entertainment" to R.drawable.ic_payment,
        "Healthcare"    to R.drawable.ic_payment,
        "Education"     to R.drawable.ic_payment,
        "Grocery"       to R.drawable.ic_receipt_long,
        "Insurance"     to R.drawable.ic_payment,
        "P2P Transfer"  to R.drawable.ic_sms,
        "EMI"           to R.drawable.ic_payment,
        "Misc"          to R.drawable.ic_payment,
        "Uncategorized" to R.drawable.ic_warning,
    )

    // ── ViewHolder — holds references to item_transaction.xml views ───────────
    inner class TxnViewHolder(
        private val binding: ItemTransactionBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(txn: TransactionHistory) {

            // ── Text fields ───────────────────────────────────────────────────
            binding.tvPayee.text    = txn.payee.ifBlank { "Unknown Payee" }
            binding.tvCategory.text = txn.category
            binding.tvAmount.text   = currencyFormat.format(txn.amount)
                .replace("₹", "₹")      // Ensure proper rupee symbol
            binding.tvDate.text     = if (txn.timestamp > 0L)
                dateFormat.format(Date(txn.timestamp))
            else
                txn.date.take(6).ifBlank { "—" }

            // ── Category icon ─────────────────────────────────────────────────
            val iconRes = categoryIconMap[txn.category] ?: R.drawable.ic_payment
            binding.ivCategoryIcon.setImageResource(iconRes)

            // ── FRAUD TINTING ─────────────────────────────────────────────────
            //
            //  When isFraud=true (set by Layer 3 after FastAPI responds):
            //    1. Card background → light red (#FFF0F0)
            //    2. Card stroke     → red (#FFCDD2)
            //    3. Warning icon    → VISIBLE
            //    4. Fraud score     → VISIBLE below the amount
            //    5. Amount text     → red colour
            //  When isFraud=false: all defaults restored (white card, normal text)
            //
            if (txn.isFraud) {
                binding.cardTransaction.apply {
                    setCardBackgroundColor(
                        ContextCompat.getColor(context, R.color.fraud_red_bg)
                    )
                    strokeColor = ContextCompat.getColor(context, R.color.card_stroke_fraud)
                }
                binding.tvAmount.setTextColor(
                    ContextCompat.getColor(binding.root.context, R.color.fraud_text)
                )
                binding.ivFraudWarning.visibility = android.view.View.VISIBLE
                binding.tvFraudScore.visibility   = android.view.View.VISIBLE
                binding.tvFraudScore.text         = "Score: ${"%.0f".format(txn.fraudScore * 100)}%"
            } else {
                binding.cardTransaction.apply {
                    setCardBackgroundColor(
                        ContextCompat.getColor(context, R.color.surface_card)
                    )
                    strokeColor = ContextCompat.getColor(context, R.color.card_stroke_normal)
                }
                binding.tvAmount.setTextColor(
                    ContextCompat.getColor(binding.root.context, R.color.text_primary)
                )
                binding.ivFraudWarning.visibility = android.view.View.GONE
                binding.tvFraudScore.visibility   = android.view.View.GONE
            }

            // ── Click listener — open transaction detail sheet ────────────────
            binding.root.setOnClickListener { onItemClick(txn) }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): TxnViewHolder {
        val binding = ItemTransactionBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return TxnViewHolder(binding)
    }

    override fun onBindViewHolder(holder: TxnViewHolder, position: Int) {
        holder.bind(getItem(position))
    }
}
