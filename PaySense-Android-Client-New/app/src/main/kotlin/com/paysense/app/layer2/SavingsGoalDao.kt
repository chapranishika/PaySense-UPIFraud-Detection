package com.paysense.app.layer2

import androidx.room.*
import kotlinx.coroutines.flow.Flow

/**
 * ============================================================================
 *  SavingsGoalDao.kt  — Phase 3: Savings Goals
 *
 *  Reactive Flow-based DAO so FinanceFragment re-renders goal cards
 *  automatically whenever a goal is created, updated, or completed.
 * ============================================================================
 */
@Dao
interface SavingsGoalDao {

    /** All active (non-completed) goals ordered by creation date. */
    @Query("SELECT * FROM savings_goals WHERE isCompleted = 0 ORDER BY createdAt ASC")
    fun getActiveGoalsFlow(): Flow<List<SavingsGoal>>

    /** All goals including completed — for history view. */
    @Query("SELECT * FROM savings_goals ORDER BY isCompleted ASC, createdAt ASC")
    fun getAllGoalsFlow(): Flow<List<SavingsGoal>>

    /** One-shot fetch for pre-filling GoalBottomSheet when editing. */
    @Query("SELECT * FROM savings_goals WHERE id = :id LIMIT 1")
    suspend fun getGoalById(id: Int): SavingsGoal?

    /** Insert a new goal or replace an existing one (for edits). */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertGoal(goal: SavingsGoal)

    /**
     * Update savedAmount for a goal.
     * Called when user taps "+ Add savings" on a goal card.
     */
    @Query("UPDATE savings_goals SET savedAmount = :amount WHERE id = :id")
    suspend fun updateSavedAmount(id: Int, amount: Double)

    /**
     * Mark a goal as completed.
     * Completed goals stay in the DB for history — not deleted.
     */
    @Query("UPDATE savings_goals SET isCompleted = 1 WHERE id = :id")
    suspend fun completeGoal(id: Int)

    /** Permanently delete a goal. */
    @Delete
    suspend fun deleteGoal(goal: SavingsGoal)

    /** Count of active goals — used for empty state detection. */
    @Query("SELECT COUNT(*) FROM savings_goals WHERE isCompleted = 0")
    suspend fun getActiveGoalCount(): Int
}
