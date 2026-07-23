package com.paysense.app.layer2

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

// ============================================================================
//  PaySenseDatabase.kt  (v3 — proper migration replaces destructive wipe)
//
//  FIXED (v2 → v3):
//  v2 used fallbackToDestructiveMigration(). On every app update that bumps
//  the database version, this silently deletes all user data. For PaySense,
//  that means the entire 90-day transaction history used for z-score
//  calculation is wiped — every user instantly becomes a cold-start user
//  after an update. That is catastrophic for the personalisation feature.
//
//  v3 adds MIGRATION_1_2 which adds the transaction_history table without
//  touching the existing payee_category_cache table. Users keep their
//  full transaction history and their HITL payee mappings across updates.
//
//  Migration versioning rule: every schema change must have a corresponding
//  Migration object. Never use fallbackToDestructiveMigration() in production.
// ============================================================================
@Database(
    entities     = [PayeeCache::class, TransactionHistory::class,
                    Budget::class, SavingsGoal::class],
    version      = 5,          // v4→v5: added savings_goals table (Phase 3)
    exportSchema = true
)
abstract class PaySenseDatabase : RoomDatabase() {

    abstract fun payeeDao(): PayeeDao
    abstract fun transactionDao(): TransactionDao
    abstract fun budgetDao(): BudgetDao
    abstract fun savingsGoalDao(): SavingsGoalDao   // Phase 3

    companion object {

        // ── MIGRATION 1 → 2 ───────────────────────────────────────────────────
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("""
                    CREATE TABLE IF NOT EXISTS `transaction_history` (
                        `txnId`       TEXT    NOT NULL,
                        `payee`       TEXT    NOT NULL,
                        `amount`      REAL    NOT NULL,
                        `category`    TEXT    NOT NULL,
                        `senderId`    TEXT    NOT NULL,
                        `date`        TEXT    NOT NULL,
                        `timestamp`   INTEGER NOT NULL,
                        `fraudScore`  REAL    NOT NULL DEFAULT 0.0,
                        `isFraud`     INTEGER NOT NULL DEFAULT 0,
                        `alertLevel`  TEXT    NOT NULL DEFAULT 'none',
                        PRIMARY KEY(`txnId`)
                    )
                """.trimIndent())
            }
        }

        // ── MIGRATION 2 → 3 ───────────────────────────────────────────────────
        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL(
                    "ALTER TABLE `transaction_history` ADD COLUMN `hourOfDay` INTEGER"
                )
            }
        }

        // ── MIGRATION 3 → 4 ───────────────────────────────────────────────────
        val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("""
                    CREATE TABLE IF NOT EXISTS `budgets` (
                        `category`     TEXT    NOT NULL,
                        `monthlyLimit` REAL    NOT NULL,
                        `createdAt`    INTEGER NOT NULL,
                        PRIMARY KEY(`category`)
                    )
                """.trimIndent())
            }
        }

        // ── MIGRATION 4 → 5 ───────────────────────────────────────────────────
        //  Adds savings_goals table for Phase 3 goal tracking.
        //  Purely additive — all existing tables unaffected.
        val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("""
                    CREATE TABLE IF NOT EXISTS `savings_goals` (
                        `id`           INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        `name`         TEXT    NOT NULL,
                        `targetAmount` REAL    NOT NULL,
                        `savedAmount`  REAL    NOT NULL DEFAULT 0.0,
                        `deadline`     INTEGER,
                        `emoji`        TEXT    NOT NULL DEFAULT '🎯',
                        `createdAt`    INTEGER NOT NULL,
                        `isCompleted`  INTEGER NOT NULL DEFAULT 0
                    )
                """.trimIndent())
            }
        }

        @Volatile private var INSTANCE: PaySenseDatabase? = null

        fun getInstance(context: Context): PaySenseDatabase =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    PaySenseDatabase::class.java,
                    "paysense_local.db"
                )
                .addMigrations(MIGRATION_1_2, MIGRATION_2_3,
                               MIGRATION_3_4, MIGRATION_4_5)
                .build()
                .also { INSTANCE = it }
            }
    }
}
