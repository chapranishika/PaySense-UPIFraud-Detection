package com.paysense.app.ui

import android.view.LayoutInflater
import android.view.View
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

class TransactionAdapter(
    private val onItemClick: (TransactionHistory) -> Unit = {}
) : ListAdapter<TransactionHistory, TransactionAdapter.TxnViewHolder>(DiffCallback) {

    private object DiffCallback : DiffUtil.ItemCallback<TransactionHistory>() {
        override fun areItemsTheSame(old: TransactionHistory, new: TransactionHistory) = old.txnId == new.txnId
        override fun areContentsTheSame(old: TransactionHistory, new: TransactionHistory) = old == new
    }

    private val currencyFormat = NumberFormat.getCurrencyInstance(Locale("en", "IN"))
    private val dateFormat = SimpleDateFormat("dd MMM", Locale.getDefault())

    inner class TxnViewHolder(private val binding: ItemTransactionBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(txn: TransactionHistory) {
            binding.tvPayee.text = txn.payee.ifBlank { "Unknown" }
            binding.tvCategory.text = txn.category
            binding.tvAmount.text = currencyFormat.format(txn.amount)
            binding.tvDate.text = if (txn.timestamp > 0L) dateFormat.format(Date(txn.timestamp)) else txn.date

            if (txn.isFraud) {
                binding.cardTransaction.setCardBackgroundColor(ContextCompat.getColor(binding.root.context, R.color.fraud_red_bg))
                binding.cardTransaction.strokeColor = ContextCompat.getColor(binding.root.context, R.color.card_stroke_fraud)
                binding.tvAmount.setTextColor(ContextCompat.getColor(binding.root.context, R.color.fraud_text))
                binding.ivFraudWarning.visibility = View.VISIBLE
                binding.tvFraudScore.visibility = View.VISIBLE
                binding.tvFraudScore.text = "Score: ${(txn.fraudScore * 100).toInt()}%"
            } else {
                binding.cardTransaction.setCardBackgroundColor(ContextCompat.getColor(binding.root.context, R.color.surface_card))
                binding.cardTransaction.strokeColor = ContextCompat.getColor(binding.root.context, R.color.card_stroke_normal)
                binding.tvAmount.setTextColor(ContextCompat.getColor(binding.root.context, R.color.text_primary))
                binding.ivFraudWarning.visibility = View.GONE
                binding.tvFraudScore.visibility = View.GONE
            }
            binding.root.setOnClickListener { onItemClick(txn) }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): TxnViewHolder {
        return TxnViewHolder(ItemTransactionBinding.inflate(LayoutInflater.from(parent.context), parent, false))
    }

    override fun onBindViewHolder(holder: TxnViewHolder, position: Int) {
        holder.bind(getItem(position))
    }
}
