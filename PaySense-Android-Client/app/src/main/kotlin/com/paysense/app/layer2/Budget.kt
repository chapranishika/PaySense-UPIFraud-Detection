package com.paysense.app.layer2

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * ============================================================================
 *  Budget.kt  — Phase 2: Finance Tracker
 *
 *  Stores the user's self-set monthly spending limit per category.
 *  One row per category — upsert semantics (insert or replace).
 *
 *  SCHEMA: budgets table (added in MIGRATION_3_4)
 *    category     TEXT  PRIMARY KEY   e.g. "Food", "Travel"
 *    monthlyLimit REAL  NOT NULL      user-set limit in INR
 *    createdAt    INT   NOT NULL      epoch ms
 *
 *  Design decisions:
 *    • category is the PRIMARY KEY so there is always at most one budget
 *      per category — no duplicates, natural upsert via @Insert(onConflict=REPLACE)
 *    • monthlyLimit is in raw INR (not thousands) so arithmetic with
 *      CategorySpend.total (also raw INR) is direct and type-safe
 *    • No foreign key to transaction_history — budgets are independent
 *      of transaction existence (user can set a budget before spending)
 * ============================================================================
 */
@Entity(tableName = "budgets")
data class Budget(

    @PrimaryKey
    val category     : String,   // matches TransactionHistory.category values

    val monthlyLimit : Double,   // user's monthly limit in INR

    val createdAt    : Long = System.currentTimeMillis()
)
