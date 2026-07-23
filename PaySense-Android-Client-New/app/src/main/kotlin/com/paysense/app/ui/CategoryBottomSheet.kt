package com.paysense.app.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.activityViewModels
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.google.android.material.chip.Chip
import com.paysense.app.databinding.LayoutBottomSheetCategoryBinding
import com.paysense.app.layer2.PayeeCacheRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.text.NumberFormat
import java.util.Locale

/**
 * ============================================================================
 *  CategoryBottomSheet.kt  (v2 — MVVM fix)
 *
 *  FIX (v1 → v2):
 *    v1 accessed txnDao directly from the Fragment, bypassing the ViewModel:
 *
 *      lifecycleScope.launch {
 *          repo.saveUserCategory(payeeName, category)
 *          txnDao.updateTransactionCategory(txnId, category)  // ← WRONG
 *      }
 *
 *    This created two paths to transaction_history (ViewModel + Fragment),
 *    making the codebase architecturally inconsistent.
 *
 *  v2 uses activityViewModels() to share the Activity's ViewModel:
 *    • viewModel.updateCategory() replaces the direct txnDao call
 *    • activityViewModels() returns the SAME instance MainActivity observes
 *    • Room Flow re-emits → StateFlow updates → RecyclerView re-renders the
 *      row without any manual refresh or broadcast needed
 *
 *  WHY activityViewModels() NOT viewModels():
 *    viewModels() creates a Fragment-scoped ViewModel — a DIFFERENT instance
 *    from MainActivity's. The StateFlow the RecyclerView observes would not
 *    be updated, so the row would not re-render despite the Room write.
 *    activityViewModels() returns the Activity's existing instance — the one
 *    the RecyclerView is already collecting from.
 *
 *  ARCHITECTURAL BOUNDARY:
 *    PayeeCacheRepository.saveUserCategory() is still called directly because
 *    payee_category_cache is not owned by PaySenseViewModel — it has its own
 *    Repository. This is correct: the ViewModel owns transaction_history;
 *    the Repository owns payee_category_cache. Two tables, two owners.
 * ============================================================================
 */
class CategoryBottomSheet : BottomSheetDialogFragment() {

    companion object {
        const val TAG = "CategoryBottomSheet"
        private const val ARG_PAYEE  = "payee_name"
        private const val ARG_AMOUNT = "amount"
        private const val ARG_TXN_ID = "txn_id"

        fun newInstance(payee: String, amount: Double, txnId: String): CategoryBottomSheet =
            CategoryBottomSheet().apply {
                arguments = Bundle().apply {
                    putString(ARG_PAYEE,  payee)
                    putDouble(ARG_AMOUNT, amount)
                    putString(ARG_TXN_ID, txnId)
                }
            }
    }

    // activityViewModels() — shares the Activity's existing ViewModel instance
    private val viewModel: PaySenseViewModel by activityViewModels()

    private var _binding: LayoutBottomSheetCategoryBinding? = null
    private val binding get() = _binding!!

    private val payeeName by lazy { arguments?.getString(ARG_PAYEE)  ?: "Unknown Payee" }
    private val amount    by lazy { arguments?.getDouble(ARG_AMOUNT) ?: 0.0 }
    private val txnId     by lazy { arguments?.getString(ARG_TXN_ID) ?: "" }

    private val currencyFmt = NumberFormat.getCurrencyInstance(Locale("en", "IN"))

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = LayoutBottomSheetCategoryBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        (dialog as? BottomSheetDialog)?.behavior?.apply {
            state         = BottomSheetBehavior.STATE_EXPANDED
            skipCollapsed = true
        }

        binding.tvPromptDetail.text = "${currencyFmt.format(amount)} to $payeeName"
        setupChips()
        binding.tvSkip.setOnClickListener { dismiss() }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun setupChips() {
        val chipGroup = binding.chipGroupCategories
        for (i in 0 until chipGroup.childCount) {
            val chip = chipGroup.getChildAt(i) as? Chip ?: continue
            chip.setOnClickListener {
                val category = chip.tag as? String ?: return@setOnClickListener
                chip.isChecked = true
                onCategorySelected(category)
            }
        }
    }

    /**
     * Two writes — each to the correct architectural owner.
     *
     * Write 1 → payee_category_cache via PayeeCacheRepository
     *   This table is cross-transaction state (a permanent payee→category
     *   mapping). It belongs to the Repository layer, not the ViewModel.
     *   A plain CoroutineScope(Dispatchers.IO) is fine here because the
     *   Repository is a singleton that outlives the Fragment.
     *
     * Write 2 → transaction_history via PaySenseViewModel.updateCategory()
     *   Single source of truth for transaction state. The Room Flow
     *   auto-emits the updated list → StateFlow → RecyclerView re-renders.
     *   Runs in viewModelScope (cancelled cleanly on Activity destroy).
     */
    private fun onCategorySelected(category: String) {

        // Write 1: payee cache (Repository boundary — correct)
        CoroutineScope(Dispatchers.IO).launch {
            PayeeCacheRepository
                .getInstance(requireContext())
                .saveUserCategory(payeeName, category)
        }

        // Write 2: transaction history (ViewModel boundary — correct)
        viewModel.updateCategory(txnId, category)

        dismiss()
    }
}
