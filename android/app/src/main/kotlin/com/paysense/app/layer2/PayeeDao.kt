package com.paysense.app.layer2

import androidx.room.*
import com.paysense.app.layer2.PayeeCache

/**
 * ============================================================================
 *  PaySense — PayeeDao.kt
 *  Layer 2: Room Data Access Object — SQL interface for payee_category_cache
 *
 *  All functions are suspend functions — they MUST be called from a coroutine
 *  (never from the main thread). Room enforces this at compile time.
 *
 *  The DAO is an interface — Room generates the implementation at compile time
 *  using annotation processing. You never write the actual SQL execution code.
 * ============================================================================
 */
@Dao
interface PayeeDao {

    /**
     * Looks up the stored category for a given payee name.
     *
     * Normalises the query to lowercase before comparison so that
     * "Zomato", "ZOMATO", and "zomato" all resolve to the same cached row.
     *
     * @return The [PayeeCache] entry if found, or null on a cache miss.
     *         The caller (SmsReceiver dispatch logic) uses null to decide
     *         whether to trigger the HITL user prompt.
     */
    @Query("SELECT * FROM payee_category_cache WHERE payeeName = :payeeName LIMIT 1")
    suspend fun getCategoryForPayee(payeeName: String): PayeeCache?

    /**
     * Inserts or updates a payee-to-category mapping.
     *
     * OnConflictStrategy.REPLACE means:
     *   - First time "Ramesh" is seen → INSERT a new row.
     *   - If user later re-categorises "Ramesh" → UPDATE the existing row.
     * This keeps the table clean (no duplicates) with zero extra logic.
     *
     * @param payeeCache The entity to persist. Always normalise payeeName
     *                   to lowercase before calling this function.
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPayeeCategory(payeeCache: PayeeCache)

    /**
     * Retrieves all cached entries. Used by the Settings screen to let the
     * user review and edit their saved payee categories.
     *
     * ORDER BY updatedAt DESC → most recently seen payees appear first.
     */
    @Query("SELECT * FROM payee_category_cache ORDER BY updatedAt DESC")
    suspend fun getAllCachedPayees(): List<PayeeCache>

    /**
     * Deletes a specific payee mapping. Used when the user wants to
     * "forget" a category and be prompted again for that payee.
     */
    @Query("DELETE FROM payee_category_cache WHERE payeeName = :payeeName")
    suspend fun deletePayeeCategory(payeeName: String)

    /**
     * Returns count of all user-defined categories (source = "user").
     * Shown in the app's stats dashboard as "X payees categorised by you".
     */
    @Query("SELECT COUNT(*) FROM payee_category_cache WHERE source = 'user'")
    suspend fun countUserDefinedCategories(): Int
}
