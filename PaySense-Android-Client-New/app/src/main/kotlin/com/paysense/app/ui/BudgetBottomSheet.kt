package com.paysense.app.ui

import android.os.Bundle
import android.text.InputType
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.viewModels
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import kotlinx.coroutines.launch
import java.text.NumberFormat
import java.util.Locale

/**
 * ============================================================================
 *  BudgetBottomSheet.kt  — Phase 2: Budget Setting UI
 *
 *  Shown when the user taps "Set budget" or the budget bar on a category row.
 *
 *  FLOW:
 *    1. User taps "Set budget" on the Food row → BudgetBottomSheet.newInstance("Food")
 *    2. Sheet pre-fills any existing budget (getBudgetForCategory)
 *    3. User types ₹5,000 and taps "Save"
 *    4. viewModel.setBudget("Food", 5000.0) → upsert → Room Flow re-emits
 *    5. FinanceFragment budget progress bar for Food → updates reactively
 *    6. Sheet dismisses
 *
 *  USES FinanceViewModel:
 *    Uses viewModels() (Fragment-scoped) which returns the SAME FinanceViewModel
 *    instance as FinanceFragment because both are in the same Activity scope.
 *    Actually uses activityViewModels() variant scoped to Activity so the
 *    FinanceFragment's ViewModel is shared — same instance, same Flow.
 *
 *  DELETE BUDGET:
 *    A "Remove budget" button appears when a budget already exists,
 *    allowing the user to remove the limit without setting a new one.
 * ============================================================================
 */
class BudgetBottomSheet : BottomSheetDialogFragment() {

    companion object {
        const val TAG         = "BudgetBottomSheet"
        private const val ARG_CATEGORY     = "category"
        private const val ARG_CURRENT_LIMIT = "current_limit"

        fun newInstance(category: String, currentLimit: Double = 0.0): BudgetBottomSheet =
            BudgetBottomSheet().apply {
                arguments = Bundle().apply {
                    putString(ARG_CATEGORY,      category)
                    putDouble(ARG_CURRENT_LIMIT, currentLimit)
                }
            }
    }

    // Shares FinanceFragment's FinanceViewModel
    private val viewModel: FinanceViewModel by activityViewModels()

    private val category     by lazy { arguments?.getString(ARG_CATEGORY)      ?: "" }
    private val currentLimit by lazy { arguments?.getDouble(ARG_CURRENT_LIMIT) ?: 0.0 }
    private val currencyFmt  = NumberFormat.getCurrencyInstance(Locale("en", "IN"))

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val dp = { v: Int -> (v * resources.displayMetrics.density).toInt() }

        val root = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(android.R.color.white)
            setPadding(0, 0, 0, dp(24))
        }

        // Drag handle
        root.addView(View(requireContext()).apply {
            setBackgroundColor(android.graphics.Color.parseColor("#E2E8F0"))
            layoutParams = LinearLayout.LayoutParams(dp(36), dp(4)).apply {
                gravity  = android.view.Gravity.CENTER_HORIZONTAL
                topMargin = dp(10); bottomMargin = dp(16)
            }
        })

        // Title
        root.addView(TextView(requireContext()).apply {
            text = if (currentLimit > 0) "Edit $category Budget" else "Set $category Budget"
            textSize = 17f
            setTextColor(android.graphics.Color.parseColor("#1A2A4A"))
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setPadding(dp(20), 0, dp(20), dp(4))
        })

        // Subtitle
        root.addView(TextView(requireContext()).apply {
            text = if (currentLimit > 0)
                "Current limit: ${currencyFmt.format(currentLimit)} / month"
            else
                "How much do you want to spend on $category each month?"
            textSize = 13f
            setTextColor(android.graphics.Color.parseColor("#64748B"))
            setPadding(dp(20), 0, dp(20), dp(16))
        })

        // Divider
        root.addView(View(requireContext()).apply {
            setBackgroundColor(android.graphics.Color.parseColor("#F1F5F9"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(1)).apply {
                bottomMargin = dp(16) }
        })

        // Amount input
        val inputRow = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(20), 0, dp(20), dp(16))
            gravity = android.view.Gravity.CENTER_VERTICAL
        }
        inputRow.addView(TextView(requireContext()).apply {
            text = "₹"; textSize = 22f
            setTextColor(android.graphics.Color.parseColor("#00B4D8"))
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setPadding(0, 0, dp(8), 0)
        })
        val amountInput = EditText(requireContext()).apply {
            hint       = "0"
            textSize   = 22f
            inputType  = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            setTextColor(android.graphics.Color.parseColor("#1A2A4A"))
            background = null
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            if (currentLimit > 0) setText(currentLimit.toLong().toString())
            selectAll()
        }
        inputRow.addView(amountInput)
        inputRow.addView(TextView(requireContext()).apply {
            text = "/ month"; textSize = 13f
            setTextColor(android.graphics.Color.parseColor("#94A3B8"))
            setPadding(dp(8), 0, 0, 0)
        })
        root.addView(inputRow)

        // Quick amount chips
        val chipRow = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(16), 0, dp(16), dp(16))
        }
        listOf(1000, 2000, 5000, 10000).forEach { amount ->
            chipRow.addView(TextView(requireContext()).apply {
                text = "₹${amount/1000}k"
                textSize = 12f
                setTextColor(android.graphics.Color.parseColor("#00B4D8"))
                setBackgroundColor(android.graphics.Color.parseColor("#EDE9FE"))
                setPadding(dp(12), dp(6), dp(12), dp(6))
                val lp = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT).apply { marginEnd = dp(8) }
                layoutParams = lp
                setOnClickListener { amountInput.setText(amount.toString()) }
            })
        }
        root.addView(chipRow)

        // Save button
        root.addView(TextView(requireContext()).apply {
            text    = if (currentLimit > 0) "Update Budget" else "Save Budget"
            textSize = 15f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(android.graphics.Color.WHITE)
            setBackgroundColor(android.graphics.Color.parseColor("#00B4D8"))
            gravity  = android.view.Gravity.CENTER
            setPadding(dp(20), dp(14), dp(20), dp(14))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                marginStart = dp(20); marginEnd = dp(20); bottomMargin = dp(12)
            }
            setOnClickListener {
                val input = amountInput.text.toString().toDoubleOrNull()
                if (input == null || input <= 0) {
                    amountInput.error = "Enter a valid amount"
                    return@setOnClickListener
                }
                viewModel.setBudget(category, input)
                dismiss()
            }
        })

        // Remove budget (only shown when editing existing)
        if (currentLimit > 0) {
            root.addView(TextView(requireContext()).apply {
                text    = "Remove budget limit"
                textSize = 13f
                setTextColor(android.graphics.Color.parseColor("#E53935"))
                gravity  = android.view.Gravity.CENTER
                setPadding(dp(20), dp(8), dp(20), dp(8))
                setOnClickListener {
                    viewModel.deleteBudget(category)
                    dismiss()
                }
            })
        }

        return root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        (dialog as? BottomSheetDialog)?.behavior?.apply {
            state         = BottomSheetBehavior.STATE_EXPANDED
            skipCollapsed = true
        }
    }
}
