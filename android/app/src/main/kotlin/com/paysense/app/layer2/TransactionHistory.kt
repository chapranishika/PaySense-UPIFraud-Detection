package com.paysense.app.layer2

import androidx.room.*

// ============================================================================
//  TransactionHistory.kt
//  Room Entity — persists every parsed, gate-cleared transaction locally.
//  This table is the source of truth for the dashboard RecyclerView and
//  for the dynamic amount deviation calculation in FraudApiService.
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

    // Fraud verdict fields — populated after Layer 3 returns
    val fraudScore    : Double  = 0.0,
    val isFraud       : Boolean = false,
    val alertLevel    : String  = "none"    // "none"|"low"|"medium"|"high"
)
