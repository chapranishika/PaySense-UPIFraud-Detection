package com.paysense.app.ui

import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.text.TextUtils
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.paysense.app.layer2.FinanceExportUtil
import com.paysense.app.layer2.PdfReportGenerator
import kotlinx.coroutines.launch
import java.text.NumberFormat
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale

/**
 * ============================================================================
 *  FinanceFragment.kt  (v4 — Phase 4: Export + Budget Surplus)
 *
 *  NEW IN v4:
 *    • Export row — "Export CSV" and "Export PDF" buttons below header card
 *    • Surplus banner — shown when calculateBudgetSurplus() > 0 and at
 *      least one active goal exists, offering "Add ₹X to [Goal]"
 *
 *  SECTION ORDER:
 *    1. Header card         (total + avg/day + budget status)
 *    1.5 Export row         ← NEW: CSV / PDF buttons
 *    1.6 Surplus banner     ← NEW: conditional, only if surplus + goals exist
 *    2. Category breakdown  (spend bars + budget progress bars)
 *    3. Insights            ← NEW: MoM + pace
 *    4. Savings Goals       ← NEW: goal cards
 *    5. Monthly trend       (6-month bar chart)
 *    6. Top merchants
 *
 *  ZERO IMPACT ON FRAUD PIPELINE: no fraud classes referenced in live code.
 * ============================================================================
 */
class FinanceFragment : Fragment() {

    private val viewModel: FinanceViewModel by viewModels()
    private val currFmt = NumberFormat.getCurrencyInstance(Locale("en", "IN"))

    private val categoryColors = mapOf(
        "Food" to "#E53935", "Food & Dining" to "#E53935",
        "Travel" to "#1E88E5", "Shopping" to "#8E24AA",
        "Bills" to "#F59E0B", "Utilities" to "#F59E0B",
        "Grocery" to "#43A047", "Entertainment" to "#FB8C00",
        "Healthcare" to "#00ACC1", "Education" to "#3949AB",
        "Recharge" to "#039BE5", "EMI" to "#6D4C41",
        "Insurance" to "#546E7A", "P2P Transfer" to "#5E35B1",
        "Misc" to "#757575", "Uncategorized" to "#9E9E9E",
    )

    private val budgetStatusColor = mapOf(
        BudgetStatus.OK      to "#43A047",
        BudgetStatus.WARNING to "#F59E0B",
        BudgetStatus.OVER    to "#E53935",
        BudgetStatus.NONE    to "#CBD5E1",
    )

    private lateinit var scrollView: ScrollView
    private lateinit var container: LinearLayout

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View {
        scrollView = ScrollView(requireContext()).apply {
            setBackgroundColor(Color.parseColor("#F0F4F8"))
        }
        this.container = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 0, 0, dp(80))
        }
        scrollView.addView(this.container)
        return scrollView
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        observeViewModel()
    }

    private fun observeViewModel() {
        viewLifecycleOwner.lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { viewModel.budgetProgress.collect { renderAll() } }
                launch { viewModel.monthOverMonth.collect { renderAll() } }
                launch { viewModel.goals.collect          { renderAll() } }
            }
        }
    }

    private fun renderAll() {
        if (!isAdded) return
        container.removeAllViews()
        renderHeaderCard()
        renderExportRow()         // Phase 4
        renderSurplusBanner()     // Phase 4
        renderCategoryBreakdown()
        renderInsights()          // Phase 3
        renderSavingsGoals()      // Phase 3
        renderMonthlyTrend()
        renderTopMerchants()
    }

    // ── 1.5 EXPORT ROW (Phase 4 NEW) ──────────────────────────────────────────
    /**
     * Two side-by-side buttons: "Export CSV" and "Export PDF".
     * Both launch a share Intent.ACTION_SEND chooser — the user picks the
     * destination app (email, Drive, WhatsApp, Files).
     *
     * CSV: raw transaction list for the current month (FinanceExportUtil)
     * PDF: formatted one-page summary report (PdfReportGenerator)
     *
     * Both run on viewLifecycleOwner.lifecycleScope — cancelled cleanly if
     * the Fragment is destroyed mid-export (e.g. user navigates away fast).
     */
    private fun renderExportRow() {
        val row = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(10), dp(8), dp(10), dp(4))
        }

        row.addView(exportButton("📄  Export CSV", "#00B4D8") {
            viewLifecycleOwner.lifecycleScope.launch {
                val txns = viewModel.getCurrentMonthTransactions()
                val label = currentMonthLabel().replace(" ", "_")
                val intent = FinanceExportUtil.exportToCsv(requireContext(), txns, label)
                if (intent != null) {
                    startActivity(Intent.createChooser(intent, "Export transactions"))
                } else {
                    showExportError()
                }
            }
        })

        row.addView(exportButton("📊  Export PDF", "#2D1B69") {
            viewLifecycleOwner.lifecycleScope.launch {
                val intent = PdfReportGenerator.generateMonthlyReport(
                    context        = requireContext(),
                    monthLabel     = currentMonthLabel(),
                    monthTotal     = viewModel.monthTotal.value,
                    avgDaily       = viewModel.avgDailySpend.value,
                    mom            = viewModel.monthOverMonth.value,
                    pace           = viewModel.spendingPace.value,
                    budgetRows     = viewModel.budgetProgress.value,
                    topMerchants   = viewModel.topMerchants.value
                )
                if (intent != null) {
                    startActivity(Intent.createChooser(intent, "Share monthly report"))
                } else {
                    showExportError()
                }
            }
        })

        container.addView(row, matchLp())
    }

    private fun exportButton(label: String, color: String, onClick: () -> Unit): TextView =
        TextView(requireContext()).apply {
            text = label; textSize = 12f
            setTextColor(Color.parseColor(color))
            setBackgroundColor(Color.parseColor("#FFFFFF"))
            gravity = android.view.Gravity.CENTER
            setPadding(dp(12), dp(10), dp(12), dp(10))
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginEnd = dp(4); marginStart = dp(4)
            }
            setOnClickListener { onClick() }
        }

    private fun showExportError() {
        Toast.makeText(requireContext(),
            "Export failed — please try again", Toast.LENGTH_SHORT).show()
    }

    private fun currentMonthLabel(): String =
        SimpleDateFormat("MMMM yyyy", Locale("en", "IN")).format(Calendar.getInstance().time)

    // ── 1.6 BUDGET SURPLUS BANNER (Phase 4 NEW) ───────────────────────────────
    /**
     * Shown only when:
     *   1. calculateBudgetSurplus() > 0  (user is under budget somewhere)
     *   2. At least one active savings goal exists
     *
     * Tapping "Add to [Goal name]" calls viewModel.applySurplusToGoal(goalId)
     * — a manual, one-time action (see FinanceViewModel docstring for why
     * this is not automatic).
     *
     * If multiple goals exist, surplus is offered to the FIRST active goal
     * (oldest by createdAt, per getActiveGoalsFlow() ordering). A future
     * enhancement could let the user pick which goal.
     */
    private fun renderSurplusBanner() {
        val surplus = viewModel.calculateBudgetSurplus()
        val goals   = viewModel.goals.value
        if (surplus <= 0 || goals.isEmpty()) return

        val targetGoal = goals.first()
        val card = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            setBackgroundColor(Color.parseColor("#EDE9FE"))
            setPadding(dp(14), dp(12), dp(14), dp(12))
        }

        val textBlock = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        textBlock.addView(tv2("💰 You're under budget by ${currFmt.format(surplus)}",
            12f, "#00B4D8", bold = true))
        textBlock.addView(tv2("Move it to ${targetGoal.emoji} ${targetGoal.name}?",
            11f, "#64748B", topPad = 2))
        card.addView(textBlock)

        card.addView(TextView(requireContext()).apply {
            text = "Add"
            textSize = 12f; typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#00B4D8"))
            setPadding(dp(16), dp(8), dp(16), dp(8))
            setOnClickListener {
                viewModel.applySurplusToGoal(targetGoal.id)
                Toast.makeText(requireContext(),
                    "${currFmt.format(surplus)} added to ${targetGoal.name}",
                    Toast.LENGTH_SHORT).show()
            }
        })

        container.addView(card, cardLp())
    }

    // ── 1. HEADER CARD ─────────────────────────────────────────────────────────
    private fun renderHeaderCard() {
        val card = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#2D1B69"))
            setPadding(dp(16), dp(16), dp(16), dp(20))
        }
        tv(card, "This Month", 11f, "#A89CC8")
        tv(card, currFmt.format(viewModel.monthTotal.value), 30f, "#FFFFFF",
            bold = true, topPad = 2)

        // Pace badge
        viewModel.spendingPace.value?.let { pace ->
            val (paceColor, paceText) = when (pace.paceStatus) {
                PaceStatus.ON_TRACK   -> "#43A047" to "✓ On track"
                PaceStatus.OVER_PACE  -> "#E53935" to "⚑ Over pace"
                PaceStatus.NEUTRAL    -> "#F59E0B" to "~ On pace"
                PaceStatus.NO_DATA    -> "#A89CC8" to ""
            }
            if (paceText.isNotEmpty()) tv(card, paceText, 11f, paceColor, topPad = 3)
        }

        val overCount = viewModel.budgetProgress.value.count { it.status == BudgetStatus.OVER }
        if (overCount > 0) tv(card, "⚠ $overCount category over budget", 11f, "#F59E0B", topPad = 2)

        val statsRow = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(12), 0, 0)
        }
        listOf(
            Triple("Avg/day",  currFmt.format(viewModel.avgDailySpend.value), "#48CAE4"),
            Triple("Top",      viewModel.topCategory.value,                    "#00B4D8"),
            Triple("Goals",    viewModel.goals.value.size.toString(),          "#48CAE4"),
        ).forEach { (label, value, col) ->
            val pill = LinearLayout(requireContext()).apply {
                orientation = LinearLayout.VERTICAL
                setBackgroundColor(Color.parseColor("#1A0F4A"))
                setPadding(dp(10), dp(8), dp(10), dp(8))
                layoutParams = LinearLayout.LayoutParams(0,
                    LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(6) }
            }
            pill.addView(tv2(value, 13f, col, bold = true))
            pill.addView(tv2(label, 9f, "#A89CC8"))
            statsRow.addView(pill)
        }
        card.addView(statsRow)
        container.addView(card, matchLp())
    }

    // ── 2. CATEGORY BREAKDOWN + BUDGET BARS (Phase 2 unchanged) ───────────────
    private fun renderCategoryBreakdown() {
        val progress = viewModel.budgetProgress.value
        if (progress.isEmpty()) return
        addSectionHeader("Spending by Category")
        val card = buildCard()
        val total = viewModel.monthTotal.value

        progress.forEachIndexed { idx, bp ->
            val cat   = bp.categorySpend
            val pct   = if (total > 0) (cat.total / total * 100f).toFloat() else 0f
            val color = categoryColors[cat.category] ?: "#757575"
            val row   = LinearLayout(requireContext()).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(14), dp(10), dp(14), dp(8))
                setOnClickListener {
                    openBudgetSheet(cat.category, bp.budget?.monthlyLimit ?: 0.0)
                }
            }
            val nameRow = LinearLayout(requireContext()).apply {
                orientation = LinearLayout.HORIZONTAL }
            nameRow.addView(tv2(cat.category, 13f, "#1A2A4A", bold = true,
                weight = 1f))
            nameRow.addView(tv2(currFmt.format(cat.total), 13f, "#1A2A4A", bold = true))
            row.addView(nameRow)
            row.addView(buildBar(color, pct, 6, 2))

            val subRow = LinearLayout(requireContext()).apply { orientation = LinearLayout.HORIZONTAL }
            subRow.addView(tv2("%.1f%%".format(pct), 10f, color, weight = 1f))
            subRow.addView(tv2("${cat.txnCount} txn${if(cat.txnCount!=1)"s" else ""}", 10f, "#94A3B8"))
            row.addView(subRow)

            when (bp.status) {
                BudgetStatus.NONE -> row.addView(tv2("+ Set monthly budget", 11f, "#00B4D8", topPad = 2))
                else -> {
                    val bc  = budgetStatusColor[bp.status] ?: "#43A047"
                    val bPct = bp.pctUsed?.coerceIn(0f, 100f) ?: 0f
                    row.addView(buildBar(bc, bPct, 2, 2, 5))
                    val br = LinearLayout(requireContext()).apply { orientation = LinearLayout.HORIZONTAL }
                    val limitText = "${currFmt.format(cat.total)} / ${currFmt.format(bp.budget!!.monthlyLimit)}"
                    val pctText   = when (bp.status) {
                        BudgetStatus.OVER    -> "OVER BUDGET"
                        BudgetStatus.WARNING -> "%.0f%% ⚠".format(bp.pctUsed)
                        else                 -> "%.0f%%".format(bp.pctUsed)
                    }
                    br.addView(tv2(limitText, 10f, bc, weight = 1f))
                    br.addView(tv2(pctText,   10f, bc))
                    row.addView(br)
                }
            }
            card.addView(row)
            if (idx < progress.size - 1) card.addView(divider())
        }
        container.addView(card, cardLp())
    }

    // ── 3. INSIGHTS — MoM + PACE (Phase 3 NEW) ─────────────────────────────────
    private fun renderInsights() {
        val mom  = viewModel.monthOverMonth.value  ?: return
        val pace = viewModel.spendingPace.value    ?: return

        addSectionHeader("Monthly Insights")

        // MoM comparison card
        val card = buildCard()
        card.setPadding(dp(14), dp(12), dp(14), dp(14))

        // This vs last month row
        val row = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
        }

        val leftBlock = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        leftBlock.addView(tv2("This month",       10f, "#94A3B8"))
        leftBlock.addView(tv2(currFmt.format(mom.thisMonth), 18f, "#1A2A4A", bold = true))
        row.addView(leftBlock)

        // Arrow + delta
        val (arrowColor, arrowText, deltaText) = when (mom.trend) {
            MoMTrend.DOWN    -> Triple("#43A047", "↓", "%.1f%% less".format(-mom.pctChange))
            MoMTrend.UP      -> Triple("#E53935", "↑", "%.1f%% more".format(mom.pctChange))
            MoMTrend.FLAT    -> Triple("#64748B", "→", "Same as last")
            MoMTrend.NO_DATA -> Triple("#94A3B8", "—", "No data yet")
        }
        val midBlock = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            gravity = android.view.Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 0.8f)
        }
        midBlock.addView(tv2(arrowText,   22f, arrowColor, bold = true).apply {
            gravity = android.view.Gravity.CENTER })
        midBlock.addView(tv2(deltaText,   10f, arrowColor).apply {
            gravity = android.view.Gravity.CENTER })
        row.addView(midBlock)

        val rightBlock = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.VERTICAL
            gravity = android.view.Gravity.END
            layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        rightBlock.addView(tv2("Last month",            10f, "#94A3B8").apply {
            gravity = android.view.Gravity.END })
        rightBlock.addView(tv2(currFmt.format(mom.lastMonth), 18f, "#1A2A4A", bold = true).apply {
            gravity = android.view.Gravity.END })
        row.addView(rightBlock)
        card.addView(row)

        // Pace projection
        card.addView(divider().apply {
            (layoutParams as? LinearLayout.LayoutParams)?.apply {
                topMargin = dp(10); bottomMargin = dp(10) }
        })

        val paceColor = when (pace.paceStatus) {
            PaceStatus.ON_TRACK  -> "#43A047"
            PaceStatus.OVER_PACE -> "#E53935"
            PaceStatus.NEUTRAL   -> "#F59E0B"
            PaceStatus.NO_DATA   -> "#94A3B8"
        }
        val paceRow = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
        }
        paceRow.addView(tv2("📈 Projected month-end:", 12f, "#64748B",
            weight = 1f))
        paceRow.addView(tv2(currFmt.format(pace.projectedTotal), 14f, paceColor, bold = true))
        card.addView(paceRow)

        val paceSubRow = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            setPadding(0, dp(3), 0, 0)
        }
        paceSubRow.addView(tv2(
            "Day ${pace.dayOfMonth} of ${pace.daysInMonth} · ₹${pace.dailyRate.toLong()}/day",
            10f, "#94A3B8", weight = 1f))
        if (pace.paceStatus != PaceStatus.NO_DATA) {
            val statusText = when (pace.paceStatus) {
                PaceStatus.ON_TRACK  -> "✓ On track"
                PaceStatus.OVER_PACE -> "⚑ Over pace"
                else                 -> "~ Neutral"
            }
            paceSubRow.addView(tv2(statusText, 10f, paceColor))
        }
        card.addView(paceSubRow)

        // Top 3 MoM category changes
        val topChanges = viewModel.categoryMoM.value
            .filter { it.lastMonth > 0 || it.thisMonth > 0 }
            .sortedByDescending { kotlin.math.abs(it.delta) }
            .take(3)

        if (topChanges.isNotEmpty()) {
            card.addView(divider().apply {
                (layoutParams as? LinearLayout.LayoutParams)?.apply {
                    topMargin = dp(10); bottomMargin = dp(6) }
            })
            card.addView(tv2("Biggest changes vs last month", 10f, "#94A3B8",
                topPad = 0).apply { setPadding(0, 0, 0, dp(4)) })
            topChanges.forEach { change ->
                val isIncrease = change.delta > 0
                val changeColor = if (isIncrease) "#E53935" else "#43A047"
                val changeText  = if (isIncrease)
                    "+${currFmt.format(change.delta)}"
                else
                    currFmt.format(change.delta)
                val cr = LinearLayout(requireContext()).apply {
                    orientation = LinearLayout.HORIZONTAL
                    setPadding(0, dp(3), 0, dp(3))
                }
                cr.addView(tv2(change.category, 11f, "#1A2A4A", weight = 1f))
                cr.addView(tv2(changeText, 11f, changeColor, bold = true))
                card.addView(cr)
            }
        }

        container.addView(card, cardLp())
    }

    // ── 4. SAVINGS GOALS (Phase 3 NEW) ─────────────────────────────────────────
    private fun renderSavingsGoals() {
        addSectionHeader("Savings Goals")

        val goals = viewModel.goals.value

        if (goals.isEmpty()) {
            // Empty state — prompt to create first goal
            val card = buildCard()
            card.setPadding(dp(16), dp(16), dp(16), dp(16))
            card.addView(tv2("🎯 No savings goals yet", 14f, "#64748B").apply {
                gravity = android.view.Gravity.CENTER
                setPadding(0, dp(8), 0, dp(8))
            })
            card.addView(TextView(requireContext()).apply {
                text = "+ Create your first goal"
                textSize = 13f; setTextColor(Color.parseColor("#00B4D8"))
                gravity = android.view.Gravity.CENTER
                setPadding(dp(16), dp(8), dp(16), dp(8))
                setBackgroundColor(Color.parseColor("#EDE9FE"))
                setOnClickListener {
                    GoalBottomSheet.newGoal()
                        .show(parentFragmentManager, GoalBottomSheet.TAG)
                }
            })
            container.addView(card, cardLp())
            return
        }

        // Goal cards
        val card = buildCard()
        goals.forEachIndexed { idx, goal ->
            val goalRow = LinearLayout(requireContext()).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(14), dp(12), dp(14), dp(10))
                setOnClickListener {
                    GoalBottomSheet.addSavings(
                        goal.id, goal.name, goal.targetAmount, goal.savedAmount)
                        .show(parentFragmentManager, GoalBottomSheet.TAG)
                }
            }

            // Emoji + name + amount
            val topRow = LinearLayout(requireContext()).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
            }
            topRow.addView(tv2(goal.emoji, 20f, "#000000").apply {
                setPadding(0, 0, dp(8), 0) })
            val nameBlock = LinearLayout(requireContext()).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0,
                    LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            nameBlock.addView(tv2(goal.name, 13f, "#1A2A4A", bold = true))
            nameBlock.addView(tv2(
                "${currFmt.format(goal.savedAmount)} / ${currFmt.format(goal.targetAmount)}",
                10f, "#94A3B8"))
            topRow.addView(nameBlock)

            // Pct badge
            val pctText = "%.0f%%".format(goal.pctComplete)
            val pctColor = when {
                goal.pctComplete >= 100f -> "#43A047"
                goal.pctComplete >= 50f  -> "#00B4D8"
                else                     -> "#64748B"
            }
            topRow.addView(tv2(pctText, 13f, pctColor, bold = true))
            goalRow.addView(topRow)

            // Progress bar
            goalRow.addView(buildBar("#00B4D8", goal.pctComplete, 6, 4))

            // Deadline if set
            goal.deadline?.let { dl ->
                val daysLeft = ((dl - System.currentTimeMillis()) /
                               (24 * 60 * 60 * 1000)).toInt()
                val deadlineText = when {
                    daysLeft < 0  -> "Overdue by ${-daysLeft} days"
                    daysLeft == 0 -> "Due today!"
                    daysLeft <= 7 -> "$daysLeft days left"
                    else -> {
                        val cal = java.util.Calendar.getInstance().apply { timeInMillis = dl }
                        "By ${cal.get(java.util.Calendar.DAY_OF_MONTH)} " +
                        java.text.SimpleDateFormat("MMM yyyy",
                            Locale.getDefault()).format(dl)
                    }
                }
                val deadlineColor = if (daysLeft < 0) "#E53935"
                                    else if (daysLeft <= 7) "#F59E0B"
                                    else "#94A3B8"
                goalRow.addView(tv2(deadlineText, 10f, deadlineColor, topPad = 2))
            }

            card.addView(goalRow)
            if (idx < goals.size - 1) card.addView(divider())
        }

        // "+ Add goal" row at the bottom
        val addRow = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER
            setPadding(dp(14), dp(10), dp(14), dp(12))
            setOnClickListener {
                GoalBottomSheet.newGoal()
                    .show(parentFragmentManager, GoalBottomSheet.TAG)
            }
        }
        addRow.addView(tv2("+ Add another goal", 13f, "#00B4D8"))
        card.addView(divider())
        card.addView(addRow)
        container.addView(card, cardLp())
    }

    // ── 5. MONTHLY TREND ──────────────────────────────────────────────────────
    private fun renderMonthlyTrend() {
        val months = viewModel.monthlySpend.value
        if (months.isEmpty()) return
        addSectionHeader("6-Month Trend")
        val card = buildCard()
        card.setPadding(dp(14), dp(12), dp(14), dp(14))
        val maxTotal = months.maxOfOrNull { it.total } ?: 1.0
        val chartRow = LinearLayout(requireContext()).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.BOTTOM
        }
        months.forEach { month ->
            val frac = (month.total / maxTotal).toFloat()
            val col  = LinearLayout(requireContext()).apply {
                orientation = LinearLayout.VERTICAL
                gravity = android.view.Gravity.CENTER_HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(0,
                    LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    marginStart = dp(4); marginEnd = dp(4) }
            }
            col.addView(tv2("₹${(month.total/1000).toInt()}k", 8f, "#94A3B8").apply {
                gravity = android.view.Gravity.CENTER })
            col.addView(View(requireContext()).apply {
                setBackgroundColor(Color.parseColor("#00B4D8"))
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    dp((120 * frac).toInt().coerceAtLeast(4))).apply { topMargin = dp(4) }
            })
            col.addView(tv2(month.month.takeLast(5).replace("-", "/"), 8f, "#64748B").apply {
                gravity = android.view.Gravity.CENTER
                setPadding(0, dp(4), 0, 0) })
            chartRow.addView(col)
        }
        card.addView(chartRow)
        container.addView(card, cardLp())
    }

    // ── 6. TOP MERCHANTS ──────────────────────────────────────────────────────
    private fun renderTopMerchants() {
        val merchants = viewModel.topMerchants.value
        if (merchants.isEmpty()) return
        addSectionHeader("Top Merchants This Month")
        val card = buildCard()
        merchants.forEachIndexed { idx, m ->
            val color = categoryColors[m.category] ?: "#757575"
            val row   = LinearLayout(requireContext()).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(dp(14), dp(12), dp(14), dp(12))
                gravity = android.view.Gravity.CENTER_VERTICAL
            }
            row.addView(tv2("${idx+1}", 11f, color).apply {
                val s = dp(30); gravity = android.view.Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(s, s).apply { marginEnd = dp(12) }
            })
            val info = LinearLayout(requireContext()).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0,
                    LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            info.addView(tv2(m.payee, 13f, "#1A2A4A", bold = true).apply {
                maxLines = 1; ellipsize = TextUtils.TruncateAt.END })
            info.addView(tv2("${m.txnCount} visit${if(m.txnCount!=1)"s" else ""} · ${m.category}",
                10f, "#94A3B8"))
            row.addView(info)
            row.addView(tv2(currFmt.format(m.total), 13f, "#1A2A4A", bold = true))
            card.addView(row)
            if (idx < merchants.size - 1) card.addView(divider())
        }
        container.addView(card, cardLp())
    }

    // ── HELPERS ───────────────────────────────────────────────────────────────
    private fun openBudgetSheet(category: String, currentLimit: Double) {
        BudgetBottomSheet.newInstance(category, currentLimit)
            .show(parentFragmentManager, BudgetBottomSheet.TAG)
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun tv(parent: LinearLayout, text: String, size: Float,
                   color: String, bold: Boolean = false, topPad: Int = 0) {
        parent.addView(tv2(text, size, color, bold, topPad))
    }

    private fun tv2(text: String, size: Float, color: String,
                    bold: Boolean = false, topPad: Int = 0,
                    weight: Float = -1f): TextView =
        TextView(requireContext()).apply {
            this.text = text; textSize = size
            setTextColor(Color.parseColor(color))
            if (bold) typeface = Typeface.DEFAULT_BOLD
            if (topPad > 0) setPadding(0, dp(topPad), 0, 0)
            if (weight >= 0) layoutParams = LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, weight)
        }

    private fun buildCard() = LinearLayout(requireContext()).apply {
        orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.WHITE) }

    private fun addSectionHeader(title: String) {
        container.addView(TextView(requireContext()).apply {
            text = title.uppercase(); textSize = 11f
            setTextColor(Color.parseColor("#64748B"))
            typeface = Typeface.DEFAULT_BOLD
            setPadding(dp(14), dp(14), dp(14), dp(6))
            letterSpacing = 0.05f
        })
    }

    private fun buildBar(color: String, pct: Float,
                         topMargin: Int = 0, bottomMargin: Int = 0,
                         height: Int = 6): FrameLayout {
        val bg = FrameLayout(requireContext()).apply {
            setBackgroundColor(Color.parseColor("#EDE9FE"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(height)).apply {
                this.topMargin = dp(topMargin); this.bottomMargin = dp(bottomMargin) }
        }
        val fill = View(requireContext()).apply {
            setBackgroundColor(Color.parseColor(color))
            layoutParams = FrameLayout.LayoutParams(1, FrameLayout.LayoutParams.MATCH_PARENT)
        }
        bg.addView(fill)
        bg.post {
            val w = (bg.width * pct / 100f).toInt()
            fill.layoutParams = FrameLayout.LayoutParams(
                maxOf(w, dp(4)), FrameLayout.LayoutParams.MATCH_PARENT)
        }
        return bg
    }

    private fun divider() = View(requireContext()).apply {
        setBackgroundColor(Color.parseColor("#F1F5F9"))
        layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 1)
    }

    private fun matchLp() = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)

    private fun cardLp() = LinearLayout.LayoutParams(
        LinearLayout.LayoutParams.MATCH_PARENT,
        LinearLayout.LayoutParams.WRAP_CONTENT).apply {
        marginStart = dp(10); marginEnd = dp(10); bottomMargin = dp(4) }

    companion object { fun newInstance() = FinanceFragment() }
}
