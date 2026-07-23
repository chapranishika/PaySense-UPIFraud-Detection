package com.paysense.app.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.paysense.app.layer2.PaySenseDatabase
import com.paysense.app.layer2.TransactionHistory
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import java.util.Calendar

/**
 * ============================================================================
 *  PaySenseViewModel.kt
 *  MVVM ViewModel — replaces direct Room queries in MainActivity.
 *
 *  WHY THIS EXISTS (the rotation bug we fixed):
 *
 *  BEFORE (buggy):
 *    MainActivity.refreshDashboard() called txnDao.getAllTransactions()
 *    inside lifecycleScope.launch {}. When the user rotated the phone:
 *      1. Android destroyed the Activity (and its lifecycleScope).
 *      2. The running Room coroutine was cancelled mid-flight.
 *      3. The new Activity created a fresh lifecycleScope and re-queried.
 *      4. For a split second the UI was blank.
 *    Worse: any in-flight Room write (e.g. updateFraudVerdict) that was
 *    cancelled mid-rotation left a stale fraudScore=0.0 in the database.
 *
 *  AFTER (fixed):
 *    The ViewModel survives rotation — Android retains it across the
 *    Activity lifecycle. The Flow<List<TransactionHistory>> from Room
 *    keeps emitting to the ViewModel's StateFlow regardless of how many
 *    times the Activity is created or destroyed. The new Activity simply
 *    re-collects from the already-running StateFlow.
 *
 *  ARCHITECTURE SUMMARY:
 *    Room DB → Flow<List<TransactionHistory>>   (auto-emits on any DB change)
 *         → ViewModel.transactions StateFlow     (survives rotation)
 *         → MainActivity collects                (re-subscribes after rotation)
 *         → RecyclerView.submitList()            (only re-renders changed rows)
 *
 *  Using AndroidViewModel (not ViewModel) because we need applicationContext
 *  to access the Room singleton. Never use Activity context in a ViewModel —
 *  it would leak the Activity even after it is destroyed.
 * ============================================================================
 */
class PaySenseViewModel(application: Application) : AndroidViewModel(application) {

    private val txnDao = PaySenseDatabase
        .getInstance(application)
        .transactionDao()

    // ── Transaction list — the primary UI data source ────────────────────────
    //
    //  _transactions is private and mutable (only this ViewModel writes to it).
    //  transactions is the public read-only StateFlow the UI observes.
    //  This is the standard Kotlin StateFlow encapsulation pattern.
    //
    //  Initial value: emptyList() — the UI renders an empty state immediately
    //  while Room loads data asynchronously.
    private val _transactions = MutableStateFlow<List<TransactionHistory>>(emptyList())
    val transactions: StateFlow<List<TransactionHistory>> = _transactions.asStateFlow()

    // ── Dashboard stats ────────────────────────────────────────────────────
    private val _totalSpent = MutableStateFlow(0.0)
    val totalSpent: StateFlow<Double> = _totalSpent.asStateFlow()

    private val _monthlyTxnCount = MutableStateFlow(0)
    val monthlyTxnCount: StateFlow<Int> = _monthlyTxnCount.asStateFlow()

    private val _fraudCount = MutableStateFlow(0)
    val fraudCount: StateFlow<Int> = _fraudCount.asStateFlow()

    // ── Loading and error state ────────────────────────────────────────────
    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    // ── Start observing Room as soon as the ViewModel is created ────────────
    //
    //  viewModelScope is automatically cancelled when the ViewModel is cleared
    //  (i.e. when the user navigates away permanently, not on rotation).
    //  This means we never leak a coroutine or hold a stale DB connection.
    init {
        observeTransactions()
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  OBSERVE TRANSACTIONS (Room Flow → StateFlow)
    //
    //  getAllTransactionsFlow() returns a Room Flow that:
    //    - Emits immediately with the current DB contents.
    //    - Re-emits automatically whenever any row changes (INSERT, UPDATE, DELETE).
    //    - Never completes unless the DAO or DB is closed.
    //
    //  We collect this Flow in viewModelScope so it runs as long as the
    //  ViewModel is alive. On each emission we:
    //    1. Update the transactions list StateFlow (feeds the RecyclerView).
    //    2. Recompute all dashboard stats (total spent, count, fraud count).
    //    3. Set isLoading=false after the first emission.
    // ──────────────────────────────────────────────────────────────────────────
    private fun observeTransactions() {
        viewModelScope.launch {
            txnDao.getAllTransactionsFlow()
                .catch { e ->
                    // Room only throws here for schema errors or DB corruption,
                    // not for normal empty states. Log and expose to the UI.
                    _error.value = "Database error: ${e.message}"
                    _isLoading.value = false
                }
                .collect { txnList ->
                    _transactions.value = txnList
                    updateDashboardStats(txnList)
                    _isLoading.value = false
                }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  UPDATE DASHBOARD STATS
    //
    //  Called every time the transaction list changes. Computes stats from the
    //  already-fetched list rather than making additional Room queries.
    //  This is more efficient than three separate DAO calls.
    // ──────────────────────────────────────────────────────────────────────────
    private fun updateDashboardStats(txnList: List<TransactionHistory>) {
        val monthStart = Calendar.getInstance().apply {
            set(Calendar.DAY_OF_MONTH, 1)
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis

        // Total spent this calendar month
        _totalSpent.value = txnList
            .filter { it.timestamp >= monthStart }
            .sumOf { it.amount }

        // Transaction count this month
        _monthlyTxnCount.value = txnList
            .count { it.timestamp >= monthStart }

        // Fraud alert count (all-time, not just this month — shows total risk)
        _fraudCount.value = txnList.count { it.isFraud }
    }

    /**
     * Called by CategoryBottomSheet after the user selects a category.
     * Updates the transaction's category in Room — the Flow will automatically
     * re-emit with the updated list, and the UI will re-render that row.
     */
    fun updateCategory(txnId: String, category: String) {
        viewModelScope.launch {
            txnDao.updateTransactionCategory(txnId, category)
            // No need to manually refresh — the Room Flow re-emits automatically.
        }
    }

    fun addManualTransaction(
        payee: String,
        amount: Double,
        category: String,
        app: String,
        device: String,
        hour: Int,
        newDevice: Int,
        ipMismatch: Int
    ) {
        viewModelScope.launch {
            val txnId = "TXN" + System.currentTimeMillis()
            val dateFormat = java.text.SimpleDateFormat("dd-MMM-yy", java.util.Locale.getDefault())
            val dateStr = dateFormat.format(java.util.Date())

            val parsed = com.paysense.app.layer1.ParsedTransaction(
                senderId  = "AD-${app.uppercase()}",
                rawBody   = "Manually entered transaction of ₹$amount to $payee via $app",
                amount    = amount,
                payee     = payee,
                txnId     = txnId,
                date      = dateStr,
                timestamp = System.currentTimeMillis()
            )

            try {
                val service = com.paysense.app.layer3.FraudApiService.getInstance(getApplication())
                service.scoreTransaction(
                    txn = parsed,
                    category = category,
                    newDeviceFlag = newDevice,
                    ipLocationMismatch = ipMismatch,
                    deviceType = device
                )
            } catch (e: Exception) {
                _error.value = "Failed to score transaction: ${e.message}"
            }
        }
    }

    /**
     * Clears any error state once the UI has shown the error to the user.
     */
    fun clearError() {
        _error.value = null
    }
}
