package com.paysense.app.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import com.paysense.app.R
import com.paysense.app.databinding.FragmentDashboardBinding
import com.paysense.app.layer2.TransactionHistory
import kotlinx.coroutines.launch
import java.text.NumberFormat
import java.util.Calendar
import java.util.Locale
import kotlin.math.roundToInt

/**
 * ============================================================================
 *  DashboardContentFragment.kt  — Phase 1: Finance tracker refactor
 *
 *  WHY THIS EXISTS:
 *    With the addition of FinanceFragment as a second tab, the dashboard
 *    content (RecyclerView + summary card) needed to move from a direct
 *    Activity layout into its own Fragment so both tabs can coexist in the
 *    same fragment_container view.
 *
 *  SAFE MIGRATION FROM MAINACTIVITY:
 *    All ViewModel observation logic is identical to the MainActivity v2
 *    approach — repeatOnLifecycle(STARTED), same StateFlow collection,
 *    same adapter.submitList() calls. The only change is that these
 *    observations now run in the Fragment's lifecycleScope rather than
 *    the Activity's. This is architecturally better: observations are
 *    scoped to the Fragment's lifecycle, so they pause when the Finance
 *    tab is showing (Fragment is hidden but not stopped in our show/hide
 *    strategy — this is fine as both tabs remain STARTED while visible).
 *
 *  VIEWMODEL SHARING:
 *    Uses activityViewModels() to share PaySenseViewModel with MainActivity
 *    and CategoryBottomSheet — same instance, same StateFlow, consistent state.
 * ============================================================================
 */
class DashboardContentFragment : Fragment() {

    // Share the Activity's PaySenseViewModel — same instance as MainActivity
    private val viewModel: PaySenseViewModel by activityViewModels()

    private var _binding: FragmentDashboardBinding? = null
    private val binding get() = _binding!!

    private lateinit var adapter: TransactionAdapter
    private val currencyFmt = NumberFormat.getCurrencyInstance(Locale("en", "IN"))
    private val currencyFmtNoDecimals = NumberFormat.getCurrencyInstance(Locale("en", "IN")).apply {
        maximumFractionDigits = 0
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentDashboardBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupRecyclerView()
        observeViewModel()
        displayRandomQuote()

        binding.fabAddTransaction.setOnClickListener {
            AddTransactionDialog.newInstance()
                .show(parentFragmentManager, AddTransactionDialog.TAG)
        }
    }

    private fun displayRandomQuote() {
        val quotes = arrayOf(
            "Do not save what is left after spending, but spend what is left after saving." to "Warren Buffett",
            "A penny saved is a penny earned." to "Benjamin Franklin",
            "Beware of little expenses; a small leak will sink a great ship." to "Benjamin Franklin",
            "An investment in knowledge pays the best interest." to "Benjamin Franklin",
            "Money is a terrible master but an excellent servant." to "P.T. Barnum",
            "Wealth consists not in having great possessions, but in having few wants." to "Epictetus"
        )
        val selected = quotes.random()
        binding.tvDailyQuote.text = "\"${selected.first}\""
        binding.tvDailyQuoteAuthor.text = "— ${selected.second}"
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    /**
     * Drives the wireframe "Security Status" card + stat tiles row entirely
     * from the real transaction list — no placeholder numbers:
     *   - Safety % / ring fill = share of ALL-TIME transactions NOT flagged
     *     as fraud (isFraud == false). 100% when there is no history yet.
     *   - "Today" tile = count of transactions with a timestamp within the
     *     current calendar day.
     *   - Status label/colour flips from green "SAFE" to amber/red "REVIEW
     *     NEEDED" the moment any transaction carries a non-"none" alertLevel
     *     — "high" wins over "medium" for the headline colour/copy.
     */
    private fun updateSecurityStatus(txnList: List<TransactionHistory>) {
        val total = txnList.size
        val fraudCount = txnList.count { it.isFraud }
        val safetyPct = if (total == 0) 100 else (100.0 * (total - fraudCount) / total).roundToInt()

        val todayStart = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis
        val todayCount = txnList.count { it.timestamp >= todayStart }

        binding.tvSecurityGaugeLabel.text = "$safetyPct%"
        binding.tvStatSafety.text = "$safetyPct%"
        binding.tvStatTxnToday.text = todayCount.toString()
        binding.gaugeSecurityStatus.progress = safetyPct / 100f

        val hasHighAlert = txnList.any { it.alertLevel == "high" }
        val ctx = requireContext()
        when {
            fraudCount == 0 -> {
                val color = ContextCompat.getColor(ctx, R.color.colorAlertNone)
                binding.gaugeSecurityStatus.setProgressColor(color)
                binding.tvSecurityStatusLabel.setTextColor(color)
                binding.tvSecurityStatusLabel.text = "SAFE"
                binding.tvSecurityStatusSubtext.text = "No suspicious activity"
            }
            hasHighAlert -> {
                val color = ContextCompat.getColor(ctx, R.color.colorAlertHigh)
                binding.gaugeSecurityStatus.setProgressColor(color)
                binding.tvSecurityStatusLabel.setTextColor(color)
                binding.tvSecurityStatusLabel.text = "AT RISK"
                binding.tvSecurityStatusSubtext.text = "$fraudCount transaction(s) need review"
            }
            else -> {
                val color = ContextCompat.getColor(ctx, R.color.colorAlertMedium)
                binding.gaugeSecurityStatus.setProgressColor(color)
                binding.tvSecurityStatusLabel.setTextColor(color)
                binding.tvSecurityStatusLabel.text = "REVIEW"
                binding.tvSecurityStatusSubtext.text = "$fraudCount transaction(s) flagged for review"
            }
        }
    }

    private fun setupRecyclerView() {
        adapter = TransactionAdapter(onItemClick = { txn ->
            TransactionDetailBottomSheet.newInstance(txn.txnId)
                .show(parentFragmentManager, TransactionDetailBottomSheet.TAG)
        })
        binding.rvTransactions.apply {
            layoutManager = LinearLayoutManager(requireContext())
            adapter        = this@DashboardContentFragment.adapter
        }
    }

    private fun observeViewModel() {

        // Transaction list → RecyclerView (+ Security Status card / stat tiles,
        // which are derived from this same real list — see updateSecurityStatus)
        viewLifecycleOwner.lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.transactions.collect { txnList ->
                    adapter.submitList(txnList)
                    binding.layoutEmptyState.visibility =
                        if (txnList.isEmpty()) View.VISIBLE else View.GONE
                    updateSecurityStatus(txnList)
                }
            }
        }

        // Total spent → summary card
        viewLifecycleOwner.lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.totalSpent.collect { total ->
                    binding.tvTotalSpent.text = currencyFmt.format(total)
                    binding.tvStatSpentTile.text = currencyFmtNoDecimals.format(total)
                }
            }
        }

        // Monthly transaction count pill
        viewLifecycleOwner.lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.monthlyTxnCount.collect { count ->
                    binding.tvTxnCount.text = count.toString()
                }
            }
        }

        // Fraud count pill
        viewLifecycleOwner.lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.fraudCount.collect { count ->
                    binding.tvFraudCount.text = count.toString()
                }
            }
        }
    }

    companion object {
        fun newInstance() = DashboardContentFragment()
    }
}
