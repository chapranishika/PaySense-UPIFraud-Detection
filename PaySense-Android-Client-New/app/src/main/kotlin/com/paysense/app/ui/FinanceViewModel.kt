package com.paysense.app.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.paysense.app.layer2.Budget
import com.paysense.app.layer2.CategorySpend
import com.paysense.app.layer2.MerchantSpend
import com.paysense.app.layer2.MonthlySpend
import com.paysense.app.layer2.MonthlyCashFlow
import com.paysense.app.layer2.PaySenseDatabase
import com.paysense.app.layer2.SavingsGoal
import com.paysense.app.layer2.TransactionHistory
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.Calendar

/**
 * ============================================================================
 *  FinanceViewModel.kt  (v3 — Phase 3: MoM insights + savings goals)
 *
 *  NEW IN v3:
 *    • monthOverMonth: MoMInsight — this month vs last month total
 *    • categoryMoM: List<CategoryMoM> — per-category delta
 *    • spendingPace: SpendingPace — projected end-of-month total
 *    • goals: List<SavingsGoal> — active savings goals (reactive Flow)
 *    • addGoal() / updateGoalSavings() / completeGoal() / deleteGoal()
 *
 *  REACTIVE PIPELINE (Phase 2 unchanged + Phase 3 additions):
 *    budgets Flow    ─┐
 *                     ├─ recomputeBudgetProgress() → budgetProgress StateFlow
 *    category spend  ─┘
 *
 *    goals Flow → goals StateFlow  (independent reactive stream)
 *
 *    MoM + Pace are one-shot (loaded once, refreshable via loadInsights())
 *
 *  ZERO IMPACT ON FRAUD PIPELINE: read-only + new savings_goals table only.
 * ============================================================================
 */
class FinanceViewModel(application: Application) : AndroidViewModel(application) {

    private val txnDao    = PaySenseDatabase.getInstance(application).transactionDao()
    private val budgetDao = PaySenseDatabase.getInstance(application).budgetDao()
    private val goalDao   = PaySenseDatabase.getInstance(application).savingsGoalDao()

    // ── Time helpers ──────────────────────────────────────────────────────────
    private fun monthStart(): Long = Calendar.getInstance().apply {
        set(Calendar.DAY_OF_MONTH, 1)
        set(Calendar.HOUR_OF_DAY, 0); set(Calendar.MINUTE, 0)
        set(Calendar.SECOND, 0);      set(Calendar.MILLISECOND, 0)
    }.timeInMillis

    private fun lastMonthStart(): Long = Calendar.getInstance().apply {
        set(Calendar.DAY_OF_MONTH, 1)
        set(Calendar.HOUR_OF_DAY, 0); set(Calendar.MINUTE, 0)
        set(Calendar.SECOND, 0);      set(Calendar.MILLISECOND, 0)
        add(Calendar.MONTH, -1)
    }.timeInMillis

    private fun sixMonthsAgo(): Long =
        System.currentTimeMillis() - (180L * 24 * 60 * 60 * 1000)

    private fun daysInCurrentMonth(): Int =
        Calendar.getInstance().getActualMaximum(Calendar.DAY_OF_MONTH)

    private fun currentDayOfMonth(): Int =
        Calendar.getInstance().get(Calendar.DAY_OF_MONTH)

    // ── Phase 1 + 2 StateFlows (unchanged) ───────────────────────────────────
    private val _monthlySpend   = MutableStateFlow<List<MonthlySpend>>(emptyList())
    val monthlySpend: StateFlow<List<MonthlySpend>> = _monthlySpend.asStateFlow()

    private val _monthlyCashFlow = MutableStateFlow<List<MonthlyCashFlow>>(emptyList())
    val monthlyCashFlow: StateFlow<List<MonthlyCashFlow>> = _monthlyCashFlow.asStateFlow()

    private val _topMerchants   = MutableStateFlow<List<MerchantSpend>>(emptyList())
    val topMerchants: StateFlow<List<MerchantSpend>> = _topMerchants.asStateFlow()

    private val _monthTotal     = MutableStateFlow(0.0)
    val monthTotal: StateFlow<Double> = _monthTotal.asStateFlow()

    private val _totalEarnings  = MutableStateFlow(0.0)
    val totalEarnings: StateFlow<Double> = _totalEarnings.asStateFlow()

    private val _netBalance     = MutableStateFlow(0.0)
    val netBalance: StateFlow<Double> = _netBalance.asStateFlow()

    private val _avgDailySpend  = MutableStateFlow(0.0)
    val avgDailySpend: StateFlow<Double> = _avgDailySpend.asStateFlow()

    private val _topCategory    = MutableStateFlow("")
    val topCategory: StateFlow<String> = _topCategory.asStateFlow()

    private val _budgetProgress = MutableStateFlow<List<BudgetProgress>>(emptyList())
    val budgetProgress: StateFlow<List<BudgetProgress>> = _budgetProgress.asStateFlow()

    // ── Phase 3 StateFlows ────────────────────────────────────────────────────
    private val _monthOverMonth = MutableStateFlow<MoMInsight?>(null)
    val monthOverMonth: StateFlow<MoMInsight?> = _monthOverMonth.asStateFlow()

    private val _categoryMoM    = MutableStateFlow<List<CategoryMoM>>(emptyList())
    val categoryMoM: StateFlow<List<CategoryMoM>> = _categoryMoM.asStateFlow()

    private val _spendingPace   = MutableStateFlow<SpendingPace?>(null)
    val spendingPace: StateFlow<SpendingPace?> = _spendingPace.asStateFlow()

    private val _goals          = MutableStateFlow<List<SavingsGoal>>(emptyList())
    val goals: StateFlow<List<SavingsGoal>> = _goals.asStateFlow()

    private val _isLoading      = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private var lastBudgets: List<Budget> = emptyList()

    init {
        viewModelScope.launch {
            PaySenseDatabase.seedMockDataIfEmpty(PaySenseDatabase.getInstance(application))
            observeTransactions()
            observeBudgetProgress()
            observeGoals()
            loadStaticData()
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  REACTIVE: Budget progress (Phase 2 — unchanged)
    // ──────────────────────────────────────────────────────────────────────────
    private fun observeTransactions() {
        viewModelScope.launch {
            txnDao.getAllTransactionsFlow().collect {
                loadStaticData()
                recomputeBudgetProgress(lastBudgets)
            }
        }
    }

    private fun observeBudgetProgress() {
        viewModelScope.launch {
            budgetDao.getAllBudgetsFlow().collect { budgets ->
                lastBudgets = budgets
                recomputeBudgetProgress(budgets)
            }
        }
    }

    private suspend fun recomputeBudgetProgress(budgets: List<Budget>) {
        val allCategories = txnDao.getCategorySpend(monthStart())
        
        val spendCats = allCategories.filter { it.category != "Income" && it.category != "Refund" }
        val earnCats = allCategories.filter { it.category == "Income" || it.category == "Refund" }
        
        val totalEarn = earnCats.sumOf { it.total }
        val totalSpend = spendCats.sumOf { it.total }
        
        _totalEarnings.value = totalEarn
        _netBalance.value = totalEarn - totalSpend
        
        val budgetMap  = budgets.associateBy { it.category }
        _budgetProgress.value = spendCats.map { cat ->
            val budget = budgetMap[cat.category]
            BudgetProgress(
                categorySpend = cat,
                budget        = budget,
                pctUsed       = budget?.let {
                    ((cat.total / it.monthlyLimit * 100).toFloat()).coerceAtLeast(0f)
                },
                status        = when {
                    budget == null                            -> BudgetStatus.NONE
                    cat.total >= budget.monthlyLimit          -> BudgetStatus.OVER
                    cat.total >= budget.monthlyLimit * 0.70f -> BudgetStatus.WARNING
                    else                                      -> BudgetStatus.OK
                }
            )
        }
        _monthTotal.value    = totalSpend
        _topCategory.value   = spendCats.firstOrNull()?.category ?: "—"
        val day = currentDayOfMonth()
        _avgDailySpend.value = if (day > 0) _monthTotal.value / day else 0.0
        _isLoading.value     = false
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  REACTIVE: Savings goals (Phase 3)
    // ──────────────────────────────────────────────────────────────────────────
    private fun observeGoals() {
        viewModelScope.launch {
            goalDao.getActiveGoalsFlow().collect { _goals.value = it }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  ONE-SHOT: Monthly trend + top merchants + MoM + pace (Phase 3)
    // ──────────────────────────────────────────────────────────────────────────
    fun loadStaticData() {
        viewModelScope.launch {
            try {
                // Phase 1
                _monthlySpend.value = txnDao.getMonthlySpend(sixMonthsAgo())
                _monthlyCashFlow.value = txnDao.getMonthlyCashFlow(sixMonthsAgo())
                _topMerchants.value = txnDao.getTopMerchants(monthStart(), topN = 5)

                // Phase 3 — MoM comparison
                val now           = System.currentTimeMillis()
                val msStart       = monthStart()
                val lmStart       = lastMonthStart()
                val thisMonthTotal = txnDao.getTotalSpendBetween(msStart, now)
                val lastMonthTotal = txnDao.getTotalSpendBetween(lmStart, msStart)

                val delta = thisMonthTotal - lastMonthTotal
                val pct   = if (lastMonthTotal > 0)
                    ((delta / lastMonthTotal) * 100f).toFloat() else 0f

                _monthOverMonth.value = MoMInsight(
                    thisMonth  = thisMonthTotal,
                    lastMonth  = lastMonthTotal,
                    delta      = delta,
                    pctChange  = pct,
                    trend      = when {
                        lastMonthTotal == 0.0 -> MoMTrend.NO_DATA
                        delta < 0             -> MoMTrend.DOWN
                        delta > 0             -> MoMTrend.UP
                        else                  -> MoMTrend.FLAT
                    }
                )

                // Per-category MoM
                val thisCats  = txnDao.getCategorySpendBetween(msStart, now)
                    .associateBy { it.category }
                val lastCats  = txnDao.getCategorySpendBetween(lmStart, msStart)
                    .associateBy { it.category }
                val allCats   = (thisCats.keys + lastCats.keys).toSet()
                _categoryMoM.value = allCats.map { cat ->
                    val t = thisCats[cat]?.total ?: 0.0
                    val l = lastCats[cat]?.total ?: 0.0
                    CategoryMoM(
                        category    = cat,
                        thisMonth   = t,
                        lastMonth   = l,
                        delta       = t - l,
                        pctChange   = if (l > 0) ((t - l) / l * 100f).toFloat() else 0f
                    )
                }.sortedByDescending { it.thisMonth }

                // Spending pace projection
                val dayOfMonth   = currentDayOfMonth()
                val daysInMonth  = daysInCurrentMonth()
                val spentSoFar   = txnDao.getTotalSpendSince(msStart)
                val dailyRate    = if (dayOfMonth > 0) spentSoFar / dayOfMonth else 0.0
                val projected    = dailyRate * daysInMonth

                _spendingPace.value = SpendingPace(
                    spentSoFar   = spentSoFar,
                    projectedTotal = projected,
                    dailyRate    = dailyRate,
                    dayOfMonth   = dayOfMonth,
                    daysInMonth  = daysInMonth,
                    paceStatus   = when {
                        lastMonthTotal == 0.0     -> PaceStatus.NO_DATA
                        projected <= lastMonthTotal * 0.90 -> PaceStatus.ON_TRACK
                        projected <= lastMonthTotal * 1.10 -> PaceStatus.NEUTRAL
                        else                       -> PaceStatus.OVER_PACE
                    }
                )

            } catch (_: Exception) { /* graceful degradation */ }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  BUDGET CRUD (Phase 2 — unchanged)
    // ──────────────────────────────────────────────────────────────────────────
    fun setBudget(category: String, monthlyLimit: Double) {
        if (monthlyLimit <= 0) return
        viewModelScope.launch {
            budgetDao.upsertBudget(Budget(category = category, monthlyLimit = monthlyLimit))
        }
    }

    fun deleteBudget(category: String) {
        viewModelScope.launch { budgetDao.deleteBudget(category) }
    }

    suspend fun getBudgetForCategory(category: String) =
        budgetDao.getBudgetForCategory(category)

    // ──────────────────────────────────────────────────────────────────────────
    //  GOAL CRUD (Phase 3)
    // ──────────────────────────────────────────────────────────────────────────
    fun addGoal(name: String, targetAmount: Double,
                deadline: Long? = null, emoji: String = "🎯") {
        if (name.isBlank() || targetAmount <= 0) return
        viewModelScope.launch {
            goalDao.upsertGoal(
                SavingsGoal(name = name, targetAmount = targetAmount,
                            deadline = deadline, emoji = emoji)
            )
        }
    }

    /** Add amount to a goal's savedAmount. */
    fun addSavings(goalId: Int, additionalAmount: Double) {
        viewModelScope.launch {
            val goal = goalDao.getGoalById(goalId) ?: return@launch
            val newTotal = (goal.savedAmount + additionalAmount)
                .coerceAtMost(goal.targetAmount)
            goalDao.updateSavedAmount(goalId, newTotal)
            // Auto-complete if goal is fully funded
            if (newTotal >= goal.targetAmount) {
                goalDao.completeGoal(goalId)
            }
        }
    }

    fun completeGoal(goalId: Int) {
        viewModelScope.launch { goalDao.completeGoal(goalId) }
    }

    fun deleteGoal(goal: SavingsGoal) {
        viewModelScope.launch { goalDao.deleteGoal(goal) }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  PHASE 4: EXPORT + BUDGET SURPLUS
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Returns all non-fraud transactions for the current month, suitable for
     * CSV export. Called by FinanceFragment, which passes the result to
     * FinanceExportUtil.exportToCsv().
     *
     * Read-only — does not modify transaction_history.
     */
    suspend fun getCurrentMonthTransactions(): List<TransactionHistory> {
        return txnDao.getTransactionsBetween(monthStart(), System.currentTimeMillis())
    }

    /**
     * Computes total "budget surplus" — the sum, across all budgeted
     * categories, of (monthlyLimit - spent) for categories currently
     * UNDER budget. Categories with no budget or over budget contribute 0.
     *
     * Example: Food budget ₹5,000, spent ₹3,200 → surplus ₹1,800
     *          Travel budget ₹2,000, spent ₹2,500 → surplus ₹0 (over, not negative)
     *          Shopping: no budget set → surplus ₹0 (not counted)
     *          Total surplus = ₹1,800
     *
     * This represents money the user "saved" relative to their own plans —
     * a natural amount to suggest moving into a savings goal.
     *
     * Returns 0.0 if no budgets are set (nothing to compute surplus against).
     */
    fun calculateBudgetSurplus(): Double {
        return _budgetProgress.value
            .filter { it.budget != null && it.status != BudgetStatus.OVER }
            .sumOf { bp ->
                val limit = bp.budget!!.monthlyLimit
                val spent = bp.categorySpend.total
                (limit - spent).coerceAtLeast(0.0)
            }
    }

    /**
     * Applies the current budget surplus to a savings goal's savedAmount.
     * This is a ONE-TIME manual action triggered by the user tapping
     * "Add surplus to goal" — NOT an automatic background process.
     *
     * Rationale for manual trigger (not automatic):
     *   Automatically moving money between "budgets" and "goals" without
     *   explicit user action could be surprising or undesirable — the user
     *   may have under-spent this month because they're traveling next month
     *   and need that buffer. Phase 4 keeps the user in control; full
     *   automation is a documented Phase 5 idea (with an opt-in toggle).
     *
     * Delegates to the existing addSavings() — same auto-complete logic
     * applies if the surplus completes the goal.
     */
    fun applySurplusToGoal(goalId: Int) {
        val surplus = calculateBudgetSurplus()
        if (surplus <= 0) return
        addSavings(goalId, surplus)
    }
}

// ── Phase 2 data classes (unchanged) ──────────────────────────────────────────
data class BudgetProgress(
    val categorySpend : CategorySpend,
    val budget        : Budget?,
    val pctUsed       : Float?,
    val status        : BudgetStatus
)

enum class BudgetStatus { NONE, OK, WARNING, OVER }

// ── Phase 3 data classes ───────────────────────────────────────────────────────

data class MoMInsight(
    val thisMonth  : Double,
    val lastMonth  : Double,
    val delta      : Double,   // positive = spending more, negative = spending less
    val pctChange  : Float,
    val trend      : MoMTrend
)

enum class MoMTrend { UP, DOWN, FLAT, NO_DATA }

data class CategoryMoM(
    val category  : String,
    val thisMonth : Double,
    val lastMonth : Double,
    val delta     : Double,
    val pctChange : Float
)

data class SpendingPace(
    val spentSoFar    : Double,
    val projectedTotal: Double,
    val dailyRate     : Double,
    val dayOfMonth    : Int,
    val daysInMonth   : Int,
    val paceStatus    : PaceStatus
)

enum class PaceStatus { ON_TRACK, NEUTRAL, OVER_PACE, NO_DATA }
