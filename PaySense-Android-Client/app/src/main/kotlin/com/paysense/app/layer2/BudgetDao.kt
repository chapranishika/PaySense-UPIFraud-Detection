package com.paysense.app.layer2

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

/**
 * ============================================================================
 *  BudgetDao.kt  — Phase 2: Finance Tracker
 *
 *  Provides CRUD access to the budgets table.
 *
 *  KEY DESIGN:
 *    getAllBudgetsFlow() returns a Flow so FinanceViewModel can observe
 *    budget changes reactively — when the user sets or edits a budget in
 *    BudgetBottomSheet, the FinanceFragment re-renders budget progress bars
 *    automatically without any manual refresh.
 *
 *    upsertBudget() uses OnConflictStrategy.REPLACE so "set budget" and
 *    "edit budget" are the same operation — no separate update query needed.
 * ============================================================================
 */
@Dao
interface BudgetDao {

    /**
     * Reactive stream of all budgets.
     * Room re-emits whenever any row in the budgets table changes.
     * FinanceViewModel merges this with CategorySpend to build BudgetProgress.
     */
    @Query("SELECT * FROM budgets ORDER BY category ASC")
    fun getAllBudgetsFlow(): Flow<List<Budget>>

    /**
     * One-shot lookup for a specific category budget.
     * Used by BudgetBottomSheet to pre-fill the existing limit when editing.
     */
    @Query("SELECT * FROM budgets WHERE category = :category LIMIT 1")
    suspend fun getBudgetForCategory(category: String): Budget?

    /**
     * Insert or replace — handles both "set new budget" and "edit existing".
     * OnConflictStrategy.REPLACE deletes the old row and inserts the new one,
     * which triggers Room's Flow observers on getAllBudgetsFlow().
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertBudget(budget: Budget)

    /**
     * Remove a budget limit for a category.
     * After deletion, FinanceFragment shows "Set budget" for that row.
     */
    @Query("DELETE FROM budgets WHERE category = :category")
    suspend fun deleteBudget(category: String)

    /** Count of categories with an active budget — used for empty state. */
    @Query("SELECT COUNT(*) FROM budgets")
    suspend fun getBudgetCount(): Int
}
