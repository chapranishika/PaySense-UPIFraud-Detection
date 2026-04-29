package com.paysense.app.layer2

import androidx.room.*
import com.paysense.app.layer2.TransactionHistory

/**
 * ============================================================================
 *  TransactionDao.kt
 *  DAO for transaction_history.
 *
 *  STDDEV NOTE:
 *  SQLite has no native STDDEV() aggregate function. We work around this
 *  by fetching raw amounts for a rolling 90-day window, then computing the
 *  population standard deviation in Kotlin:
 *
 *    σ = sqrt( (1/N) * Σ(xᵢ - μ)² )
 *
 *  This is intentionally population std dev (not sample, i.e. N not N-1)
 *  because we want to describe the user's known transaction distribution,
 *  not estimate a larger population. For typical users with 20-100
 *  transactions in 90 days, the difference is negligible.
 * ============================================================================
 */
@Dao
interface TransactionDao {

    // ── Writes ───────────────────────────────────────────────────────────────

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTransaction(txn: TransactionHistory)

    @Query("UPDATE transaction_history SET fraudScore=:score, isFraud=:isFraud, alertLevel=:level WHERE txnId=:txnId")
    suspend fun updateFraudVerdict(txnId: String, score: Double, isFraud: Boolean, level: String)

    /** Called by CategoryBottomSheet when the user selects a category in the HITL prompt. */
    @Query("UPDATE transaction_history SET category=:category WHERE txnId=:txnId")
    suspend fun updateTransactionCategory(txnId: String, category: String)

    // ── Dashboard reads ───────────────────────────────────────────────────────

    /** All transactions ordered newest-first — feeds the RecyclerView. */
    @Query("SELECT * FROM transaction_history ORDER BY timestamp DESC")
    suspend fun getAllTransactions(): List<TransactionHistory>

    /** Total debited this calendar month — feeds the "Total Spent" card. */
    @Query("""
        SELECT COALESCE(SUM(amount), 0.0)
        FROM   transaction_history
        WHERE  timestamp >= :monthStartMs
    """)
    suspend fun getTotalSpentSince(monthStartMs: Long): Double

    // ── Stats for dynamic deviation calculation ───────────────────────────────

    /**
     * Returns all transaction amounts from the past [windowDays] days.
     * Fetched as a raw list so Kotlin can compute variance/stddev.
     * A 90-day window balances recency with enough samples for a
     * stable standard deviation (at least 5–10 transactions expected).
     */
    @Query("""
        SELECT amount
        FROM   transaction_history
        WHERE  timestamp >= :sinceMs
        ORDER  BY timestamp DESC
    """)
    suspend fun getAmountsSince(sinceMs: Long): List<Double>

    /** Simple average — used as a fast scalar when stddev is not needed. */
    @Query("SELECT AVG(amount) FROM transaction_history WHERE timestamp >= :sinceMs")
    suspend fun getAverageAmountSince(sinceMs: Long): Double?

    /** Count of transactions in the rolling window — guards against empty DB. */
    @Query("SELECT COUNT(*) FROM transaction_history WHERE timestamp >= :sinceMs")
    suspend fun getTransactionCountSince(sinceMs: Long): Int
}
