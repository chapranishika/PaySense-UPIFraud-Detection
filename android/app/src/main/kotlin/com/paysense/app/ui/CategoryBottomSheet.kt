package com.paysense.app.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.google.android.material.chip.Chip
import com.paysense.app.R
import com.paysense.app.databinding.LayoutBottomSheetCategoryBinding
import com.paysense.app.layer2.PaySenseDatabase
import com.paysense.app.layer2.PayeeCacheRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.NumberFormat
import java.util.Locale

/**
 * ============================================================================
 *  CategoryBottomSheet.kt
 *  Human-in-the-Loop (HITL) category selection — Layer 2
 *
 *  Triggered by:  SmsReceiver broadcasting "com.paysense.SHOW_CATEGORY_PROMPT"
 *  Listened by:   MainActivity.categoryPromptReceiver
 *
 *  Flow:
 *    1. User sees: "What was your payment of ₹150 to Ramesh Auto for?"
 *    2. User taps a category chip (e.g., "Travel").
 *    3. Kotlin saves PayeeCache(payeeName="ramesh auto", category="Travel",
 *       source="user") to Room via PayeeCacheRepository.
 *    4. Also updates the TransactionHistory row so the dashboard list
 *       immediately reflects the new category without a reload.
 *    5. Bottom sheet dismisses itself.
 *    6. Next time "Ramesh Auto" appears → Tier 1 cache HIT → never prompted again.
 *
 *  Design:
 *    Extends BottomSheetDialogFragment (Material 3) — handles back-press,
 *    drag-to-dismiss, dim overlay, and keyboard avoidance automatically.
 * ============================================================================
 */
class CategoryBottomSheet : BottomSheetDialogFragment() {

    companion object {
        const val TAG = "CategoryBottomSheet"

        private const val ARG_PAYEE  = "payee_name"
        private const val ARG_AMOUNT = "amount"
        private const val ARG_TXN_ID = "txn_id"

        /**
         * Factory function — always use this instead of the constructor.
         * Arguments are passed via Bundle so they survive process death and
         * fragment back-stack restoration.
         */
        fun newInstance(payee: String, amount: Double, txnId: String): CategoryBottomSheet =
            CategoryBottomSheet().apply {
                arguments = Bundle().apply {
                    putString(ARG_PAYEE,  payee)
                    putDouble(ARG_AMOUNT, amount)
                    putString(ARG_TXN_ID, txnId)
                }
            }
    }

    // ── View binding — avoids findViewById() entirely ────────────────────────
    private var _binding: LayoutBottomSheetCategoryBinding? = null
    private val binding get() = _binding!!

    // ── Arguments from SmsReceiver via MainActivity ───────────────────────────
    private val payeeName by lazy { arguments?.getString(ARG_PAYEE)  ?: "Unknown Payee" }
    private val amount    by lazy { arguments?.getDouble(ARG_AMOUNT) ?: 0.0 }
    private val txnId     by lazy { arguments?.getString(ARG_TXN_ID) ?: "" }

    private val currencyFmt = NumberFormat.getCurrencyInstance(Locale("en", "IN"))

    // ──────────────────────────────────────────────────────────────────────────
    //  Lifecycle
    // ──────────────────────────────────────────────────────────────────────────
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        _binding = LayoutBottomSheetCategoryBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // ── Expand bottom sheet fully on launch (don't half-expand) ──────────
        (dialog as? BottomSheetDialog)?.behavior?.apply {
            state         = BottomSheetBehavior.STATE_EXPANDED
            skipCollapsed = true
        }

        setupPromptText()
        setupChips()
        setupSkipButton()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null     // Prevent memory leak from the binding holding a View reference
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  UI Setup
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Populates the dynamic prompt text with the actual payee name and amount.
     * Example output: "₹1,500.00 to Ramesh Auto"
     */
    private fun setupPromptText() {
        val formattedAmount = currencyFmt.format(amount)
        binding.tvPromptDetail.text = "$formattedAmount to $payeeName"
    }

    /**
     * Attaches a single click listener to the entire ChipGroup.
     *
     * We iterate the ChipGroup's children to find the tapped chip, then
     * read its android:tag attribute — which we set to the category string
     * in the XML (e.g., tag="Food", tag="Travel") — to get the category name
     * without hardcoding the IDs in Kotlin.
     *
     * This approach means adding a new category only requires adding a new
     * Chip in the XML — no Kotlin changes needed.
     */
    private fun setupChips() {
        val chipGroup = binding.chipGroupCategories

        for (i in 0 until chipGroup.childCount) {
            val chip = chipGroup.getChildAt(i) as? Chip ?: continue

            chip.setOnClickListener {
                val category = chip.tag as? String ?: return@setOnClickListener

                // Visual feedback: highlight tapped chip before dismissing
                chip.isChecked = true

                // Persist to Room and dismiss
                onCategorySelected(category)
            }
        }
    }

    private fun setupSkipButton() {
        binding.tvSkip.setOnClickListener {
            // User chose not to categorise right now.
            // The transaction stays as "Uncategorized" in Room.
            // The HITL prompt WILL appear again next time this payee is seen
            // (because we do NOT write to the cache on skip).
            dismiss()
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Core HITL logic — save category and dismiss
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Called when the user taps a chip.
     *
     * Performs two Room writes:
     *   1. PayeeCache  → permanent payee-to-category mapping (Tier 1 cache)
     *   2. TransactionHistory → updates the current transaction's category
     *      so the dashboard row shows the correct label immediately.
     *
     * Both writes happen in a coroutine on the IO dispatcher.
     * The sheet is dismissed immediately for a snappy UX feel — the DB
     * writes complete silently in the background.
     */
    private fun onCategorySelected(category: String) {
        val repo   = PayeeCacheRepository.getInstance(requireContext())
        val txnDao = PaySenseDatabase.getInstance(requireContext()).transactionDao()

        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                // ── Write 1: Payee cache — prevents future HITL prompts ───────────
                repo.saveUserCategory(payeeName, category)

                // ── Write 2: Update current transaction's category in history ──────
                txnDao.updateTransactionCategory(txnId, category)
            }
        }

        // Dismiss immediately
        dismiss()
    }
}
