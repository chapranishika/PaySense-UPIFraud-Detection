package com.paysense.app.layer2

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import java.util.Calendar
import java.util.Locale

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

        // ── MIGRATION_1_2 ... MIGRATION_4_5 ──
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
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

        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    "ALTER TABLE `transaction_history` ADD COLUMN `hourOfDay` INTEGER"
                )
            }
        }

        val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS `budgets` (
                        `category`     TEXT    NOT NULL,
                        `monthlyLimit` REAL    NOT NULL,
                        `createdAt`    INTEGER NOT NULL,
                        PRIMARY KEY(`category`)
                    )
                """.trimIndent())
            }
        }

        val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("""
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

        suspend fun seedMockDataIfEmpty(database: PaySenseDatabase) {
            val dao = database.transactionDao()
            if (dao.getTransactionCount() <= 5) {
                val templates = listOf(
                    // Current Month
                    MockTxn("Salary", 50000.0, "Income", 0, false, "none"),
                    MockTxn("Starbucks Coffee", 150.0, "Food & Dining", 0, false, "none"),
                    MockTxn("Uber Ride", 450.0, "Travel", 0, false, "none"),
                    MockTxn("Amazon Store", 3200.0, "Shopping", 0, false, "none"),
                    MockTxn("Netflix Premium", 199.0, "Entertainment", 0, false, "none"),
                    MockTxn("Reliance Fresh", 1200.0, "Grocery", 0, false, "none"),
                    
                    // Month -1
                    MockTxn("Salary Credit", 50000.0, "Income", -1, false, "none"),
                    MockTxn("McDonalds Burger", 350.0, "Food & Dining", -1, false, "none"),
                    MockTxn("Uber Ride", 520.0, "Travel", -1, false, "none"),
                    MockTxn("H&M Apparel", 2200.0, "Shopping", -1, false, "none"),
                    MockTxn("Refund GPay", 450.0, "Refund", -1, false, "none"),
                    MockTxn("Tata Power Bill", 1800.0, "Bills", -1, false, "none"),
                    MockTxn("Supermarket Grocery", 1500.0, "Grocery", -1, false, "none"),
                    
                    // Month -2
                    MockTxn("Salary Credit", 50000.0, "Income", -2, false, "none"),
                    MockTxn("Zomato Delivery", 850.0, "Food & Dining", -2, false, "none"),
                    MockTxn("IndiGo Airlines", 4500.0, "Travel", -2, false, "none"),
                    MockTxn("Myntra Shopping", 2800.0, "Shopping", -2, false, "none"),
                    MockTxn("BookMyShow Movie", 600.0, "Entertainment", -2, false, "none"),
                    
                    // Month -3
                    MockTxn("Salary Credit", 50000.0, "Income", -3, false, "none"),
                    MockTxn("Pizza Hut", 1200.0, "Food & Dining", -3, false, "none"),
                    MockTxn("Uber Trip", 380.0, "Travel", -3, false, "none"),
                    MockTxn("House Rent", 12000.0, "Bills", -3, false, "none"),
                    
                    // Month -4
                    MockTxn("Salary Credit", 50000.0, "Income", -4, false, "none"),
                    MockTxn("Starbucks Coffee", 250.0, "Food & Dining", -4, false, "none"),
                    MockTxn("Uber Trip", 420.0, "Travel", -4, false, "none"),
                    MockTxn("Amazon India", 1500.0, "Shopping", -4, false, "none"),
                    MockTxn("Netflix Premium", 199.0, "Entertainment", -4, false, "none")
                )
                
                templates.forEachIndexed { index, t ->
                    val cal = Calendar.getInstance().apply {
                        add(Calendar.MONTH, t.monthOffset)
                        set(Calendar.DAY_OF_MONTH, 5 + (index * 3) % 20)
                        set(Calendar.HOUR_OF_DAY, 9 + (index % 10))
                        set(Calendar.MINUTE, (index * 7) % 60)
                        set(Calendar.SECOND, 0)
                        set(Calendar.MILLISECOND, 0)
                    }
                    val dateFmt = java.text.SimpleDateFormat("dd-MMM-yy", Locale.US)
                    val dateStr = dateFmt.format(cal.time)
                    val id = "MOCK" + (cal.timeInMillis + index)
                    
                    val entity = TransactionHistory(
                        txnId = id,
                        payee = t.payee,
                        amount = t.amount,
                        category = t.category,
                        senderId = if (t.category == "Income") "AD-HDFCBK" else "AD-GPAY",
                        date = dateStr,
                        timestamp = cal.timeInMillis,
                        hourOfDay = cal.get(Calendar.HOUR_OF_DAY),
                        fraudScore = 0.02,
                        isFraud = false,
                        alertLevel = "none"
                    )
                    dao.insertTransaction(entity)
                }
            }
        }
    }
}

private data class MockTxn(
    val payee: String,
    val amount: Double,
    val category: String,
    val monthOffset: Int,
    val isFraud: Boolean,
    val alertLevel: String
)
