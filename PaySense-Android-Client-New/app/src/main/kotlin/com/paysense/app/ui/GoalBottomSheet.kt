package com.paysense.app.ui

import android.os.Bundle
import android.text.InputType
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.activityViewModels
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import java.text.NumberFormat
import java.util.Locale

/**
 * ============================================================================
 *  GoalBottomSheet.kt  — Phase 3: Create / add-to savings goal
 *
 *  MODE 1 — Create new goal (goalId = -1, default):
 *    User enters name, target amount, optional emoji, taps "Create Goal".
 *    viewModel.addGoal() upserts to Room → goals Flow re-emits → UI updates.
 *
 *  MODE 2 — Add savings to existing goal (goalId >= 0):
 *    User enters amount to add. "Add Savings" button calls
 *    viewModel.addSavings(goalId, amount). If target is met, goal is
 *    auto-completed and disappears from the active list.
 *
 *  MODE 3 — Mark complete / delete (goalId >= 0, showActions = true):
 *    Shows "Mark as complete" and "Delete goal" options.
 * ============================================================================
 */
class GoalBottomSheet : BottomSheetDialogFragment() {

    companion object {
        const val TAG = "GoalBottomSheet"
        private const val ARG_GOAL_ID      = "goal_id"
        private const val ARG_GOAL_NAME    = "goal_name"
        private const val ARG_GOAL_TARGET  = "goal_target"
        private const val ARG_GOAL_SAVED   = "goal_saved"

        /** Open sheet to create a NEW goal. */
        fun newGoal(): GoalBottomSheet = GoalBottomSheet().apply {
            arguments = Bundle().apply { putInt(ARG_GOAL_ID, -1) }
        }

        /** Open sheet to add savings to an EXISTING goal. */
        fun addSavings(goalId: Int, goalName: String,
                       targetAmount: Double, savedAmount: Double): GoalBottomSheet =
            GoalBottomSheet().apply {
                arguments = Bundle().apply {
                    putInt(ARG_GOAL_ID,     goalId)
                    putString(ARG_GOAL_NAME,   goalName)
                    putDouble(ARG_GOAL_TARGET, targetAmount)
                    putDouble(ARG_GOAL_SAVED,  savedAmount)
                }
            }
    }

    private val viewModel: FinanceViewModel by activityViewModels()
    private val currFmt = NumberFormat.getCurrencyInstance(Locale("en", "IN"))

    private val goalId     by lazy { arguments?.getInt(ARG_GOAL_ID, -1)     ?: -1 }
    private val goalName   by lazy { arguments?.getString(ARG_GOAL_NAME)    ?: "" }
    private val goalTarget by lazy { arguments?.getDouble(ARG_GOAL_TARGET)  ?: 0.0 }
    private val goalSaved  by lazy { arguments?.getDouble(ARG_GOAL_SAVED)   ?: 0.0 }

    private val isNewGoal: Boolean get() = goalId < 0

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        val dp: (Int) -> Int = { (it * resources.displayMetrics.density).toInt() }

        val root = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(android.R.color.white)
            setPadding(0, 0, 0, dp(24))
        }

        // Handle
        root.addView(View(requireContext()).apply {
            setBackgroundColor(android.graphics.Color.parseColor("#E2E8F0"))
            layoutParams = LinearLayout.LayoutParams(dp(36), dp(4)).apply {
                gravity = android.view.Gravity.CENTER_HORIZONTAL
                topMargin = dp(10); bottomMargin = dp(16)
            }
        })

        if (isNewGoal) buildCreateMode(root, dp)
        else           buildAddSavingsMode(root, dp)

        return root
    }

    // ── MODE 1: Create new goal ───────────────────────────────────────────────
    private fun buildCreateMode(root: LinearLayout, dp: (Int) -> Int) {
        addTitle(root, dp, "New Savings Goal")
        addSubtitle(root, dp, "What are you saving for?")

        // Emoji picker row
        val emojiRow = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(20), 0, dp(20), dp(12))
        }
        val emojis = listOf("🎯","✈️","🏠","📱","🚗","🎓","💍","🏖️","🎁","💰")
        var selectedEmoji = "🎯"
        val emojiViews = mutableListOf<TextView>()

        emojis.forEach { emoji ->
            val chip = TextView(requireContext()).apply {
                text = emoji; textSize = 20f; gravity = android.view.Gravity.CENTER
                setPadding(dp(8), dp(6), dp(8), dp(6))
                setBackgroundColor(if (emoji == selectedEmoji)
                    android.graphics.Color.parseColor("#EDE9FE")
                    else android.graphics.Color.TRANSPARENT)
                setOnClickListener {
                    selectedEmoji = emoji
                    emojiViews.forEach { it.setBackgroundColor(android.graphics.Color.TRANSPARENT) }
                    setBackgroundColor(android.graphics.Color.parseColor("#EDE9FE"))
                }
            }
            emojiViews.add(chip)
            emojiRow.addView(chip)
        }
        root.addView(emojiRow)

        // Goal name input
        val nameInput = addLabeledInput(root, dp, "Goal name", "e.g. Trip to Goa",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES)

        // Target amount input
        val amountInput = addLabeledInput(root, dp, "Target amount (₹)", "e.g. 25000",
            InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL)

        // Quick chips
        addQuickChips(root, dp, amountInput, listOf(5000, 10000, 25000, 50000))

        // Create button
        addPrimaryButton(root, dp, "Create Goal") {
            val name   = nameInput.text.toString().trim()
            val target = amountInput.text.toString().toDoubleOrNull()
            if (name.isEmpty()) { nameInput.error = "Enter a goal name"; return@addPrimaryButton }
            if (target == null || target <= 0) {
                amountInput.error = "Enter a valid amount"; return@addPrimaryButton
            }
            viewModel.addGoal(name, target, emoji = selectedEmoji)
            dismiss()
        }
    }

    // ── MODE 2: Add savings to existing goal ──────────────────────────────────
    private fun buildAddSavingsMode(root: LinearLayout, dp: (Int) -> Int) {
        addTitle(root, dp, goalName)

        val remaining = (goalTarget - goalSaved).coerceAtLeast(0.0)
        addSubtitle(root, dp,
            "${currFmt.format(goalSaved)} saved · ${currFmt.format(remaining)} to go")

        // Progress bar
        val pct = ((goalSaved / goalTarget) * 100f).toFloat().coerceIn(0f, 100f)
        val barBg = FrameLayout(requireContext()).apply {
            setBackgroundColor(android.graphics.Color.parseColor("#EDE9FE"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(8)).apply {
                marginStart = dp(20); marginEnd = dp(20)
                topMargin = dp(4); bottomMargin = dp(16)
            }
        }
        val barFill = View(requireContext()).apply {
            setBackgroundColor(android.graphics.Color.parseColor("#00B4D8"))
            layoutParams = FrameLayout.LayoutParams(1, FrameLayout.LayoutParams.MATCH_PARENT)
        }
        barBg.addView(barFill)
        barBg.post {
            val w = (barBg.width * pct / 100f).toInt()
            barFill.layoutParams = FrameLayout.LayoutParams(
                maxOf(w, dp(4)), FrameLayout.LayoutParams.MATCH_PARENT)
        }
        root.addView(barBg)

        val amountInput = addLabeledInput(root, dp, "Add savings (₹)",
            "e.g. ${remaining.toLong()}",
            InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL)

        addQuickChips(root, dp, amountInput,
            listOf(500, 1000, 2000, remaining.toLong().toInt().coerceAtLeast(1))
                .distinct().take(4))

        addPrimaryButton(root, dp, "Add Savings") {
            val amount = amountInput.text.toString().toDoubleOrNull()
            if (amount == null || amount <= 0) {
                amountInput.error = "Enter a valid amount"; return@addPrimaryButton
            }
            viewModel.addSavings(goalId, amount)
            dismiss()
        }

        // Mark complete / delete
        root.addView(View(requireContext()).apply {
            setBackgroundColor(android.graphics.Color.parseColor("#F1F5F9"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(1)).apply {
                topMargin = dp(8) }
        })
        addSecondaryButton(root, dp, "Mark as complete", "#00B4D8") {
            viewModel.completeGoal(goalId); dismiss()
        }
        addSecondaryButton(root, dp, "Delete goal", "#E53935") {
            val goal = com.paysense.app.layer2.SavingsGoal(
                id = goalId, name = goalName, targetAmount = goalTarget)
            viewModel.deleteGoal(goal); dismiss()
        }
    }

    // ── UI Helpers ────────────────────────────────────────────────────────────
    private fun addTitle(root: LinearLayout, dp: (Int) -> Int, text: String) {
        root.addView(TextView(requireContext()).apply {
            this.text = text; textSize = 17f
            setTextColor(android.graphics.Color.parseColor("#1A2A4A"))
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setPadding(dp(20), 0, dp(20), dp(4))
        })
    }

    private fun addSubtitle(root: LinearLayout, dp: (Int) -> Int, text: String) {
        root.addView(TextView(requireContext()).apply {
            this.text = text; textSize = 13f
            setTextColor(android.graphics.Color.parseColor("#64748B"))
            setPadding(dp(20), 0, dp(20), dp(14))
        })
    }

    private fun addLabeledInput(root: LinearLayout, dp: (Int) -> Int,
                                 label: String, hint: String, inputType: Int): EditText {
        root.addView(TextView(requireContext()).apply {
            text = label; textSize = 11f
            setTextColor(android.graphics.Color.parseColor("#64748B"))
            setPadding(dp(20), 0, dp(20), dp(4))
        })
        val input = EditText(requireContext()).apply {
            this.hint = hint; this.inputType = inputType; textSize = 16f
            setTextColor(android.graphics.Color.parseColor("#1A2A4A"))
            background = null
            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT)
            lp.marginStart = dp(20); lp.marginEnd = dp(20); lp.bottomMargin = dp(4)
            layoutParams = lp
        }
        root.addView(input)
        return input
    }

    private fun addQuickChips(root: LinearLayout, dp: (Int) -> Int,
                               target: EditText, amounts: List<Int>) {
        val row = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(16), dp(4), dp(16), dp(12))
        }
        amounts.forEach { amount ->
            row.addView(TextView(requireContext()).apply {
                text = if (amount >= 1000) "₹${amount/1000}k" else "₹$amount"
                textSize = 11f
                setTextColor(android.graphics.Color.parseColor("#00B4D8"))
                setBackgroundColor(android.graphics.Color.parseColor("#EDE9FE"))
                setPadding(dp(10), dp(5), dp(10), dp(5))
                val lp = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT).apply { marginEnd = dp(8) }
                layoutParams = lp
                setOnClickListener { target.setText(amount.toString()) }
            })
        }
        root.addView(row)
    }

    private fun addPrimaryButton(root: LinearLayout, dp: (Int) -> Int,
                                  label: String, onClick: () -> Unit) {
        root.addView(TextView(requireContext()).apply {
            text = label; textSize = 15f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(android.graphics.Color.WHITE)
            setBackgroundColor(android.graphics.Color.parseColor("#00B4D8"))
            gravity = android.view.Gravity.CENTER
            setPadding(dp(20), dp(14), dp(20), dp(14))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                marginStart = dp(20); marginEnd = dp(20); topMargin = dp(4); bottomMargin = dp(8)
            }
            setOnClickListener { onClick() }
        })
    }

    private fun addSecondaryButton(root: LinearLayout, dp: (Int) -> Int,
                                    label: String, color: String, onClick: () -> Unit) {
        root.addView(TextView(requireContext()).apply {
            text = label; textSize = 13f
            setTextColor(android.graphics.Color.parseColor(color))
            gravity = android.view.Gravity.CENTER
            setPadding(dp(20), dp(10), dp(20), dp(8))
            setOnClickListener { onClick() }
        })
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        (dialog as? BottomSheetDialog)?.behavior?.apply {
            state         = BottomSheetBehavior.STATE_EXPANDED
            skipCollapsed = true
        }
    }
}
