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
import androidx.recyclerview.widget.LinearLayoutManager
import com.paysense.app.databinding.FragmentDashboardBinding
import kotlinx.coroutines.launch
import java.text.NumberFormat
import java.util.Locale

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

    private fun setupRecyclerView() {
        adapter = TransactionAdapter()
        binding.rvTransactions.apply {
            layoutManager = LinearLayoutManager(requireContext())
            adapter        = this@DashboardContentFragment.adapter
        }
    }

    private fun observeViewModel() {

        // Transaction list → RecyclerView
        viewLifecycleOwner.lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.transactions.collect { txnList ->
                    adapter.submitList(txnList)
                    binding.layoutEmptyState.visibility =
                        if (txnList.isEmpty()) View.VISIBLE else View.GONE
                }
            }
        }

        // Total spent → summary card
        viewLifecycleOwner.lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.totalSpent.collect { total ->
                    binding.tvTotalSpent.text = currencyFmt.format(total)
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
