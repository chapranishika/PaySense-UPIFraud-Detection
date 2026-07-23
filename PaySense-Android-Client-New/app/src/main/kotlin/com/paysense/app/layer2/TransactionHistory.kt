package com.paysense.app.layer2

import androidx.room.*

// ============================================================================
//  TransactionHistory.kt  (v3 — stores local hourOfDay at insert time)
//
//  FIX (v2 → v3): UTC hour extraction bug
//    v2 computed hour-of-day in the DAO query from epoch timestamp.
//    This extracted UTC hours — in India (UTC+5:30), midnight IST appeared
//    as 18:30 in the data. The z-score was computed against a UTC-biased
//    mean, making the Night Owl / Early Bird personalisation wrong.
//
//    FIX: hourOfDay is now stored as a column at INSERT time, populated
//    from the device's local Calendar in FraudApiService. Always correct
//    local time regardless of timezone or daylight saving transitions.
//    getHoursSince() queries this column directly instead of computing
//    from timestamp.
//
//  Schema version: bump from 2 → 3 in PaySenseDatabase.
//  Migration: MIGRATION_2_3 adds the hourOfDay column as nullable (no default).
//  (noon — a safe neutral for existing rows that predate this column).
// ============================================================================
@Entity(tableName = "transaction_history")
data class TransactionHistory(

    @PrimaryKey
    val txnId         : String,     // UPI reference ID (unique per transaction)

    val payee         : String,     // Payee name extracted by Gate 3
    val amount        : Double,     // Transaction amount in INR
    val category      : String,     // Resolved by Layer 2 (may be "Uncategorized")
    val senderId      : String,     // TRAI sender ID (e.g., "AD-HDFCBK")
    val date          : String,     // Date string from the SMS body
    val timestamp     : Long,       // Epoch ms — used for monthly filtering

    // FIX v3: local hour stored at insert time — eliminates UTC bias in DAO.
    // Nullable (Int?) so pre-migration rows have NULL rather than a biased
    // DEFAULT value. getHoursSince() filters NULLs; computeDeviationStats()
    // treats users with insufficient non-null hours as cold-start (z=0.0).
    // treats users with insufficient non-null hours as cold-start (z=0.0).
    val hourOfDay     : Int? = null,   // 0–23 local device timezone; null = pre-migration row

    // Fraud verdict fields — populated after Layer 3 returns
    val fraudScore    : Double  = 0.0,
    val isFraud       : Boolean = false,
    val alertLevel    : String  = "none"    // "none"|"low"|"medium"|"high"
)

// ============================================================================
//  TransactionDao.kt
//  DAO for transaction_history.
//
//  STDDEV NOTE:
//  SQLite has no native STDDEV() aggregate function. We work around this
//  by fetching raw amounts for a rolling 90-day window, then computing the
//  population standard deviation in Kotlin:
//
//    σ = sqrt( (1/N) * Σ(xᵢ - μ)² )
//
//  This is intentionally population std dev (not sample, i.e. N not N-1)
//  because we want to describe the user's known transaction distribution,
//  not estimate a larger population. For typical users with 20-100
//  transactions in 90 days, the difference is negligible.
// ============================================================================
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

    /**
     * Returns the local hour-of-day (0–23) for each transaction in the window.
     * Used by FraudApiService.computeDeviationStats() to compute the user's
     * mean transaction hour and its standard deviation — enabling the
     * Night Owl / Early Bird hour-deviation z-score.
     *
     * FIX (v2 → v3):
     *   v2 computed:  CAST((timestamp / 3600000) % 24 AS INTEGER)
     *   This extracted UTC hours from epoch ms. In India (UTC+5:30), a
     *   transaction at midnight IST (00:00) appeared as 18 in this query.
     *   The user's "mean hour" was systematically shifted by 5.5 hours,
     *   making the hour deviation z-score wrong.
     *
     *   v3 queries:   hourOfDay
     *   This column is populated at INSERT time in FraudApiService from the
     *   device's local Calendar — always the correct local hour, timezone-safe.
     */
    @Query("""
        SELECT hourOfDay
        FROM   transaction_history
        WHERE  timestamp >= :sinceMs
        AND    hourOfDay IS NOT NULL
        ORDER  BY timestamp DESC
    """)
    suspend fun getHoursSince(sinceMs: Long): List<Int>

    /**
     * Flow-based version of getAllTransactions() for ViewModel observation.
     * Unlike the suspend version, this Flow stays active and emits a new
     * list automatically whenever any row in transaction_history changes
     * (insert, update, or delete). The ViewModel collects this Flow via
     * StateFlow, and the UI re-renders only when the data actually changes.
     *
     * This replaces the direct lifecycleScope.launch {} pattern in
     * MainActivity that broke on Activity rotation.
     */
    @Query("SELECT * FROM transaction_history ORDER BY timestamp DESC")
    fun getAllTransactionsFlow(): kotlinx.coroutines.flow.Flow<List<TransactionHistory>>

    // ── FINANCE TRACKER QUERIES (Phase 1) ─────────────────────────────────────

    @Query("""
        SELECT   category,
                 SUM(amount)  AS total,
                 COUNT(*)     AS txnCount
        FROM     transaction_history
        WHERE    timestamp >= :sinceMs
        AND      isFraud    = 0
        GROUP BY category
        ORDER BY total DESC
    """)
    suspend fun getCategorySpend(sinceMs: Long): List<CategorySpend>

    @Query("""
        SELECT   strftime('%Y-%m', datetime(timestamp/1000, 'unixepoch')) AS month,
                 SUM(amount)  AS total,
                 COUNT(*)     AS txnCount
        FROM     transaction_history
        WHERE    timestamp >= :sinceMs
        AND      isFraud    = 0
        GROUP BY month
        ORDER BY month ASC
    """)
    suspend fun getMonthlySpend(sinceMs: Long): List<MonthlySpend>

    @Query("""
        SELECT   payee,
                 category,
                 SUM(amount)  AS total,
                 COUNT(*)     AS txnCount
        FROM     transaction_history
        WHERE    timestamp >= :sinceMs
        AND      isFraud    = 0
        GROUP BY payee
        ORDER BY total DESC
        LIMIT    :topN
    """)
    suspend fun getTopMerchants(sinceMs: Long, topN: Int = 5): List<MerchantSpend>

    // ── PHASE 3 QUERIES ────────────────────────────────────────────────────────

    /**
     * Total non-fraud spend in an explicit start→end window.
     * Used for month-over-month comparison:
     *   thisMonth  = getTotalSpendBetween(monthStart, now)
     *   lastMonth  = getTotalSpendBetween(lastMonthStart, monthStart)
     * Returns 0.0 when no transactions exist in the window.
     */
    @Query("""
        SELECT COALESCE(SUM(amount), 0.0)
        FROM   transaction_history
        WHERE  timestamp >= :startMs
        AND    timestamp <  :endMs
        AND    isFraud    = 0
    """)
    suspend fun getTotalSpendBetween(startMs: Long, endMs: Long): Double

    /**
     * Per-category spend in an explicit window.
     * Used to compute MoM change per category:
     *   delta = thisMonthCategoryTotal - lastMonthCategoryTotal
     */
    @Query("""
        SELECT   category,
                 SUM(amount)  AS total,
                 COUNT(*)     AS txnCount
        FROM     transaction_history
        WHERE    timestamp >= :startMs
        AND      timestamp <  :endMs
        AND      isFraud    = 0
        GROUP BY category
        ORDER BY total DESC
    """)
    suspend fun getCategorySpendBetween(startMs: Long, endMs: Long): List<CategorySpend>

    /**
     * Spending pace: total non-fraud spend so far this month.
     * Used to project end-of-month spend:
     *   projectedMonthTotal = spentSoFar / dayOfMonth * daysInMonth
     */
    @Query("""
        SELECT COALESCE(SUM(amount), 0.0)
        FROM   transaction_history
        WHERE  timestamp >= :sinceMs
        AND    isFraud    = 0
    """)
    suspend fun getTotalSpendSince(sinceMs: Long): Double

    /**
     * Returns full TransactionHistory rows (not aggregates) in a window,
     * excluding fraud-flagged rows. Used by FinanceExportUtil for CSV export —
     * the only Phase 1-4 query that returns full entity rows rather than
     * an aggregate, because CSV export needs per-transaction detail.
     */
    @Query("""
        SELECT * FROM transaction_history
        WHERE  timestamp >= :startMs
        AND    timestamp <  :endMs
        AND    isFraud    = 0
        ORDER BY timestamp DESC
    """)
    suspend fun getTransactionsBetween(startMs: Long, endMs: Long): List<TransactionHistory>
}

// ── Finance Tracker data classes (Phase 1) ────────────────────────────────────
// Room @Query results that don't map to full entities use plain data classes.
// These are NOT @Entity — they have no table, just hold query result columns.

data class CategorySpend(
    val category  : String,
    val total     : Double,
    val txnCount  : Int
)

data class MonthlySpend(
    val month     : String,   // "2025-04", "2025-03", etc.
    val total     : Double,
    val txnCount  : Int
)

data class MerchantSpend(
    val payee     : String,
    val category  : String,
    val total     : Double,
    val txnCount  : Int
)
