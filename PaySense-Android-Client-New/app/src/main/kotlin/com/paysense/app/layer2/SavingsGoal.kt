package com.paysense.app.layer2

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * ============================================================================
 *  SavingsGoal.kt  — Phase 3: Savings Goals
 *
 *  A savings goal represents money the user wants to set aside for a
 *  specific purpose (e.g. "Trip to Goa ₹25,000 by Dec 2025").
 *
 *  SCHEMA: savings_goals table (added in MIGRATION_4_5)
 *    id           INTEGER  PRIMARY KEY AUTOINCREMENT
 *    name         TEXT     goal description ("Trip to Goa")
 *    targetAmount REAL     total amount needed
 *    savedAmount  REAL     amount saved so far (user-updated)
 *    deadline     INTEGER  epoch ms target date (nullable)
 *    emoji        TEXT     visual identifier ("✈️", "🏠", "📱", etc.)
 *    createdAt    INTEGER  epoch ms
 *    isCompleted  INTEGER  0|1 — completed goals are archived not deleted
 *
 *  DESIGN:
 *    savedAmount is manually updated by the user (or auto-incremented from
 *    budget surplus in a future Phase 4 feature). isCompleted allows goals
 *    to be marked done without losing the history of what was achieved.
 *    deadline is nullable — open-ended goals are valid ("save for a car").
 * ============================================================================
 */
@Entity(tableName = "savings_goals")
data class SavingsGoal(

    @PrimaryKey(autoGenerate = true)
    val id           : Int     = 0,

    val name         : String,          // user-written goal name
    val targetAmount : Double,          // total amount needed
    val savedAmount  : Double  = 0.0,   // amount saved so far
    val deadline     : Long?   = null,  // nullable epoch ms target date
    val emoji        : String  = "🎯",  // visual icon for the goal
    val createdAt    : Long    = System.currentTimeMillis(),
    val isCompleted  : Boolean = false
) {
    /** Percentage of goal achieved. Clamped to 0–100. */
    val pctComplete: Float
        get() = if (targetAmount > 0)
            ((savedAmount / targetAmount * 100).toFloat()).coerceIn(0f, 100f)
        else 0f

    /** Amount remaining to reach the goal. */
    val remaining: Double
        get() = (targetAmount - savedAmount).coerceAtLeast(0.0)
}
