package com.paysense.app.layer3

import com.google.gson.annotations.SerializedName

/**
 * ============================================================================
 *  PaySense — ApiModels.kt
 *  Layer 3: Retrofit Data Classes — matches the FastAPI Pydantic schema exactly
 *
 *  TransactionRequest  → what Android sends to POST /predict
 *  TransactionResponse → what FastAPI returns
 *
 *  @SerializedName must match the Pydantic field names in main.py character
 *  for character — a mismatch silently produces null values (Gson default).
 * ============================================================================
 */

// ──────────────────────────────────────────────────────────────────────────────
//  REQUEST — 43 model-ready features matching the FastAPI TransactionInput model
//
//  Fields marked with ? (nullable) correspond to Optional[float] in Pydantic.
//  Gson serialises null Kotlin values as JSON null, which FastAPI accepts for
//  optional fields (mrc_rating, device_risk_score, ip_risk_score).
// ──────────────────────────────────────────────────────────────────────────────
data class TransactionRequest(

    // ── Categorical fields ─────────────────────────────────────────────────
    @SerializedName("receiver_type")
    val receiverType: String,               // "Merchant" | "User"

    @SerializedName("transaction_type")
    val transactionType: String,            // "P2M" | "P2P" | "Bill Payment" | …

    @SerializedName("payment_app")
    val paymentApp: String,                 // "GPay" | "PhonePe" | "Paytm" | …

    @SerializedName("device_type")
    val deviceType: String,                 // "Android" | "iOS" | "Web"

    @SerializedName("usr_age_group")
    val usrAgeGroup: String,                // "18-24" | "25-34" | "35-44" | …

    @SerializedName("usr_preferred_app")
    val usrPreferredApp: String,

    @SerializedName("usr_preferred_device")
    val usrPreferredDevice: String,

    @SerializedName("mrc_category")
    val mrcCategory: String,               // "Food" | "Travel" | "P2P Transfer" | …

    @SerializedName("mrc_size")
    val mrcSize: String,                   // "Small" | "Medium" | "Enterprise" | "P2P"

    // ── Transaction numeric fields ─────────────────────────────────────────
    @SerializedName("amount")
    val amount: Double,

    @SerializedName("hour_of_day")
    val hourOfDay: Int,                     // 0–23

    @SerializedName("day_of_week")
    val dayOfWeek: Int,                     // 0=Monday … 6=Sunday

    @SerializedName("is_weekend")
    val isWeekend: Int,                     // 0 | 1

    @SerializedName("is_night_transaction")
    val isNightTransaction: Int,            // 1 if hour < 6 or hour ≥ 22

    // ── Velocity & behavioural signals ────────────────────────────────────
    @SerializedName("time_since_last_txn_min")
    val timeSinceLastTxnMin: Double,

    @SerializedName("transaction_velocity")
    val transactionVelocity: Double,

    @SerializedName("amount_deviation_score")
    val amountDeviationScore: Double,

    @SerializedName("failed_attempts_last_24h")
    val failedAttemptsLast24h: Double,

    @SerializedName("recurring_payment_flag")
    val recurringPaymentFlag: Int,

    @SerializedName("transaction_frequency_score")
    val transactionFrequencyScore: Double,

    // ── Security flags ─────────────────────────────────────────────────────
    @SerializedName("new_device_flag")
    val newDeviceFlag: Int,                 // 1 if unrecognised device

    @SerializedName("ip_location_mismatch")
    val ipLocationMismatch: Int,

    // ── Account-level fields ───────────────────────────────────────────────
    @SerializedName("user_city_tier")
    val userCityTier: Int,                  // 1 | 2 | 3

    @SerializedName("user_avg_monthly_txn")
    val userAvgMonthlyTxn: Double,

    @SerializedName("user_avg_txn_value")
    val userAvgTxnValue: Double,

    @SerializedName("user_loyalty_score")
    val userLoyaltyScore: Double,

    @SerializedName("balance_after_transaction")
    val balanceAfterTransaction: Double,

    @SerializedName("txn_success_flag")
    val txnSuccessFlag: Int,

    @SerializedName("kyc_verified_flag")
    val kycVerifiedFlag: Int,

    // ── User profile fields ────────────────────────────────────────────────
    @SerializedName("usr_home_city_tier")
    val usrHomeCityTier: Int,

    @SerializedName("usr_account_age_days")
    val usrAccountAgeDays: Double,

    @SerializedName("usr_linked_bank_count")
    val usrLinkedBankCount: Double,

    @SerializedName("usr_avg_monthly_txn_profile")
    val usrAvgMonthlyTxnProfile: Double,

    @SerializedName("usr_avg_txn_value_profile")
    val usrAvgTxnValueProfile: Double,

    @SerializedName("usr_is_high_risk")
    val usrIsHighRisk: Int,

    // ── Merchant profile fields ────────────────────────────────────────────
    @SerializedName("mrc_avg_daily_txn")
    val mrcAvgDailyTxn: Double,

    @SerializedName("mrc_is_registered")
    val mrcIsRegistered: Int,

    @SerializedName("mrc_rating")
    val mrcRating: Double?,                 // null for P2P transfers

    // ── Optional risk scores from threat-intel services ────────────────────
    @SerializedName("device_risk_score")
    val deviceRiskScore: Double?,           // null if not available

    @SerializedName("ip_risk_score")
    val ipRiskScore: Double?                // null if not available
)

// ──────────────────────────────────────────────────────────────────────────────
//  RESPONSE — matches PredictionResponse in FastAPI main.py
// ──────────────────────────────────────────────────────────────────────────────
data class TransactionResponse(

    @SerializedName("fraud_score")
    val fraudScore: Double,                 // Raw probability 0.0 – 1.0

    @SerializedName("is_fraud")
    val isFraud: Boolean,                   // threshold applied server-side

    @SerializedName("alert_level")
    val alertLevel: String,                 // "none" | "low" | "medium" | "high"

    @SerializedName("threshold_used")
    val thresholdUsed: Double,

    @SerializedName("model_version")
    val modelVersion: String
)
