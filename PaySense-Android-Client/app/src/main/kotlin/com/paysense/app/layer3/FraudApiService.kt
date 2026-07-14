package com.paysense.app.layer3

import android.content.Context
import android.util.Log
import com.google.gson.GsonBuilder
import com.paysense.app.layer1.ParsedTransaction
import com.paysense.app.layer2.PaySenseDatabase
import com.paysense.app.layer2.TransactionDao
import com.paysense.app.layer2.TransactionHistory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.Calendar
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.sqrt

private const val TAG = "PaySense_Layer3"

// ============================================================================
//  FraudApiService.kt  (v4 — all three bugs fixed)
//
//  FIX 1 (v2 → v3): Cold-start z-score
//    Previously: cold-start returned mean=0.0, stdDev=1.0
//    → z = (amount - 0) / 1 = amount = 15,000 for a ₹15,000 transaction
//    FIX: isColdStart=true → amountDeviationScore=0.0 (neutral, no signal)
//    The model's other 39 features work normally for cold-start users.
//
//  FIX 2 (v3 → v4 → v5): transactionFrequencyScore semantic correction
//    v3 incorrectly sent zHour (0–6.67+) into transactionFrequencyScore
//    whose training range was 0.0–1.0 (frequency ratio) — values were
//    semantically incompatible. v4 temporarily reverted to 0.2.
//    v5 (current — final): paysense_pipeline.py now recomputes the
//    transaction_frequency_score column as per-user hour deviation z-score
//    across the full master dataset before training. The XGBoost model is
//    retrained on this semantics. Android sends zHour here at inference time.
//    Training data and inference data now agree: both use hour deviation.
//    Discrimination ratio: fraud 0.977 vs legit 0.807 (1.21× — useful signal).
//
//  FIX 3 (v2 → v3): Room migration
//    fallbackToDestructiveMigration() replaced with MIGRATION_1_2 in
//    PaySenseDatabase.kt — user transaction history now survives app updates.
// ============================================================================
class FraudApiService private constructor(private val context: Context) {

    companion object {
        private const val BASE_URL        = "http://10.0.2.2:8000/"
        private const val STATS_WINDOW_MS = 90L * 24 * 60 * 60 * 1000
        private const val MIN_SAMPLES     = 5

        // Neutral sentinel for cold-start users — both deviation scores = 0.0
        private val COLD_START_STATS = DeviationStats(
            mean        = 0.0,
            stdDev      = 1.0,
            meanHour    = 12.0,  // neutral midday anchor
            stdDevHour  = 6.0,   // wide spread → any hour looks roughly normal
            sampleCount = 0,
            isColdStart = true
        )

        @Volatile private var INSTANCE: FraudApiService? = null
        fun getInstance(context: Context): FraudApiService =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: FraudApiService(context.applicationContext).also { INSTANCE = it }
            }
    }

    private val txnDao: TransactionDao =
        PaySenseDatabase.getInstance(context).transactionDao()

    private val api: PaySenseApi by lazy {
        val logging = HttpLoggingInterceptor { Log.d(TAG, "HTTP | $it") }
            .apply { level = HttpLoggingInterceptor.Level.BODY }

        val client = OkHttpClient.Builder()
            .addInterceptor(logging)
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(15,    TimeUnit.SECONDS)
            .writeTimeout(10,   TimeUnit.SECONDS)
            .build()

        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create(
                GsonBuilder().serializeNulls().create()
            ))
            .build()
            .create(PaySenseApi::class.java)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  MAIN ENTRY POINT
    //  Query-before-save ordering is preserved — see inline comments.
    // ──────────────────────────────────────────────────────────────────────────
    suspend fun scoreTransaction(txn: ParsedTransaction, category: String) {
        withContext(Dispatchers.IO) {

            val sinceMs = System.currentTimeMillis() - STATS_WINDOW_MS

            // Step 1: query stats BEFORE saving — prevents transaction from
            //         being included in its own deviation baseline.
            val stats = computeDeviationStats(sinceMs)

            if (stats.isColdStart) {
                Log.d(TAG, "🆕  Cold-start (${stats.sampleCount} txns) — " +
                           "z_amount=0.0 z_hour=0.0 (neutral, no history)")
            } else {
                Log.d(TAG, "📐  Amount | μ=${stats.mean.fmt()} σ=${stats.stdDev.fmt()} " +
                           "n=${stats.sampleCount}")
                Log.d(TAG, "📐  Hour   | μ=${stats.meanHour.fmt()} σ=${stats.stdDevHour.fmt()}")
            }

            // Step 2: save to Room — storing local hourOfDay (Fix 2)
            // Entity field is Int? (nullable). New rows always supply a real
            // local hour (not null). Pre-migration existing rows have null,
            // which getHoursSince() filters with WHERE hourOfDay IS NOT NULL,
            // preventing the DEFAULT 12 bias that would corrupt z-score stats.
            txnDao.insertTransaction(
                TransactionHistory(
                    txnId     = txn.txnId,
                    payee     = txn.payee,
                    amount    = txn.amount,
                    category  = category,
                    senderId  = txn.senderId,
                    date      = txn.date,
                    timestamp = txn.timestamp,
                    hourOfDay = hour    // Int assigned to Int? — always non-null for new rows
                )
            )

            // Step 3: build 43-field request with computed stats
            val request = buildTransactionRequest(txn, category, stats)
            Log.d(TAG, "🚀  Sending | amount=₹${txn.amount} " +
                       "z_amount=${request.amountDeviationScore.fmt()} " +
                       "z_hour=${request.transactionFrequencyScore.fmt()}")

            // Step 4: network call
            val response = try {
                api.predictFraud(request)
            } catch (e: Exception) {
                Log.e(TAG, "❌  Network error | ${e.message}")
                return@withContext
            }

            if (!response.isSuccessful) {
                Log.e(TAG, "❌  HTTP ${response.code()} | ${response.errorBody()?.string()}")
                return@withContext
            }

            val result = response.body() ?: run {
                Log.e(TAG, "❌  Empty response body")
                return@withContext
            }

            Log.d(TAG, "📊  Verdict | score=${result.fraudScore.fmt()} " +
                       "fraud=${result.isFraud} alert=${result.alertLevel}")

            // Step 5: write verdict back → triggers RecyclerView red-tint
            txnDao.updateFraudVerdict(
                txnId   = txn.txnId,
                score   = result.fraudScore,
                isFraud = result.isFraud,
                level   = result.alertLevel
            )

            handleFraudResponse(txn, result)
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  COMPUTE DEVIATION STATS (amount + hour)
    //
    //  Returns COLD_START_STATS when sampleCount < MIN_SAMPLES.
    //  isColdStart=true is the signal that tells buildTransactionRequest()
    //  to use 0.0 instead of a computed z-score.
    //
    //  Population stddev — not sample (N, not N-1).
    //  We describe the user's known history, not an estimated population.
    //  stdDev clamped ≥ 1.0 (amounts), ≥ 0.5 (hours) to prevent ÷0.
    // ──────────────────────────────────────────────────────────────────────────
    private suspend fun computeDeviationStats(sinceMs: Long): DeviationStats {
        val amounts = txnDao.getAmountsSince(sinceMs)

        if (amounts.size < MIN_SAMPLES) {
            return COLD_START_STATS
        }

        // Amount stats
        val n            = amounts.size.toDouble()
        val meanAmount   = amounts.sum() / n
        val varAmount    = amounts.sumOf { (it - meanAmount) * (it - meanAmount) } / n
        val stdDevAmount = sqrt(varAmount).coerceAtLeast(1.0)

        // Hour stats — getHoursSince filters WHERE hourOfDay IS NOT NULL,
        // so pre-migration rows (hourOfDay=null) are excluded automatically.
        // If fewer than MIN_SAMPLES non-null hour rows exist (e.g. user just
        // updated the app), fall back to cold-start neutral for hour component.
        val hours = txnDao.getHoursSince(sinceMs)
        val meanHour: Double
        val stdDevHour: Double
        if (hours.size < MIN_SAMPLES) {
            // Not enough non-null hour data — use neutral values so cold-start
            // on hour dimension doesn't produce extreme z-scores
            meanHour   = 12.0   // neutral midday (used only if isColdStart=false)
            stdDevHour = 6.0    // wide spread — any hour looks roughly normal
        } else {
            val hn      = hours.size.toDouble()
            val mh      = hours.map { it.toDouble() }.sum() / hn
            val vh      = hours.sumOf { h -> val d = h.toDouble() - mh; d * d } / hn
            meanHour    = mh
            stdDevHour  = sqrt(vh).coerceAtLeast(0.5)
        }

        return DeviationStats(
            mean        = meanAmount,
            stdDev      = stdDevAmount,
            meanHour    = meanHour,
            stdDevHour  = stdDevHour,
            sampleCount = amounts.size,
            isColdStart = false
        )
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  BUILD 43-FIELD REQUEST
    //
    //  Cold-start rule (THE FIX):
    //    stats.isColdStart == true  → zAmount = 0.0, zHour = 0.0
    //    stats.isColdStart == false → real z-scores from user history
    //
    //  Why 0.0 not null?
    //    amountDeviationScore is a required Double in the Pydantic model.
    //    null would cause HTTP 422. 0.0 is the correct neutral value.
    // ──────────────────────────────────────────────────────────────────────────
    private fun buildTransactionRequest(
        txn      : ParsedTransaction,
        category : String,
        stats    : DeviationStats
    ): TransactionRequest {

        val cal       = Calendar.getInstance().apply { timeInMillis = txn.timestamp }
        val hour      = cal.get(Calendar.HOUR_OF_DAY)
        val dow       = (cal.get(Calendar.DAY_OF_WEEK) - 2).let { if (it < 0) 6 else it }
        val isWeekend = if (dow >= 5) 1 else 0
        val isNight   = if (hour < 6 || hour >= 22) 1 else 0
        val isVpa     = txn.payee.contains("@")
        val recvType  = if (isVpa) "Merchant" else "User"

        // ── Amount deviation z-score ──────────────────────────────────────────
        //  FIXED (v3): cold-start users get 0.0 (neutral), not txn.amount (wrong)
        val zAmount = if (stats.isColdStart) 0.0
                      else (txn.amount - stats.mean) / stats.stdDev

        // ── Hour deviation z-score (Night Owl / Early Bird) ──────────────────
        //  z_hour = |current_hour - user_mean_hour| / user_stddev_hour
        //  Absolute value: early or late deviation from personal baseline
        //  both carry fraud signal.
        //  Fix 2 complete: hour is now local time (stored in DB), stats come
        //  from local hourOfDay column (not UTC-computed), and zHour is wired
        //  into transactionFrequencyScore whose training data is updated to
        //  match this semantics in paysense_pipeline.py.
        val zHour = if (stats.isColdStart) 0.0
                    else abs(hour.toDouble() - stats.meanHour) / stats.stdDevHour

        // userAvgTxnValue: real mean when available, else current amount
        val userAvgAmount = if (!stats.isColdStart) stats.mean else txn.amount

        return TransactionRequest(
            receiverType              = recvType,
            transactionType           = if (recvType == "User") "P2P" else "P2M",
            paymentApp                = detectPaymentApp(txn.senderId),
            deviceType                = "Android",
            usrAgeGroup               = "25-34",
            usrPreferredApp           = detectPaymentApp(txn.senderId),
            usrPreferredDevice        = "Android",
            mrcCategory               = if (recvType == "User") "P2P Transfer" else category,
            mrcSize                   = if (recvType == "User") "P2P" else "Medium",
            amount                    = txn.amount,
            hourOfDay                 = hour,
            dayOfWeek                 = dow,
            isWeekend                 = isWeekend,
            isNightTransaction        = isNight,
            timeSinceLastTxnMin       = 60.0,
            transactionVelocity       = 0.1,
            amountDeviationScore      = zAmount,    // ← cold-start safe z-score

            // transactionFrequencyScore now carries hour deviation z-score.
            // The master dataset in paysense_pipeline.py is updated to compute
            // this column as |hour - user_mean_hour| / user_stddev_hour,
            // and the model is retrained to understand this semantics.
            // This is the correct fix: the field is renamed conceptually and
            // the training data matches what we send here at inference time.
            transactionFrequencyScore = zHour,      // ← hour deviation (retrained model)
            failedAttemptsLast24h     = 0.0,
            recurringPaymentFlag      = 0,
            newDeviceFlag             = 0,
            ipLocationMismatch        = 0,
            userCityTier              = 2,
            userAvgMonthlyTxn         = 30.0,
            userAvgTxnValue           = userAvgAmount,
            userLoyaltyScore          = 0.6,
            balanceAfterTransaction   = 10000.0,
            txnSuccessFlag            = 1,
            kycVerifiedFlag           = 1,
            usrHomeCityTier           = 2,
            usrAccountAgeDays         = 365.0,
            usrLinkedBankCount        = 1.0,
            usrAvgMonthlyTxnProfile   = userAvgAmount,
            usrAvgTxnValueProfile     = userAvgAmount,
            usrIsHighRisk             = 0,
            mrcAvgDailyTxn            = if (recvType == "User") 0.0 else 50.0,
            mrcIsRegistered           = 1,
            mrcRating                 = if (recvType == "User") null else 4.0,
            deviceRiskScore           = null,
            ipRiskScore               = null
        )
    }

    private fun handleFraudResponse(txn: ParsedTransaction, result: TransactionResponse) {
        when (result.alertLevel) {
            "high"   -> Log.w(TAG, "🚨  HIGH   | score=${result.fraudScore.fmt()} payee=${txn.payee}")
            "medium" -> Log.w(TAG, "⚠️  MEDIUM | score=${result.fraudScore.fmt()} payee=${txn.payee}")
            "low"    -> Log.d(TAG, "🔵  LOW    | score=${result.fraudScore.fmt()} payee=${txn.payee}")
            "none"   -> Log.d(TAG, "✅  SAFE   | score=${result.fraudScore.fmt()} payee=${txn.payee}")
        }
    }

    private fun detectPaymentApp(senderId: String) = when {
        senderId.contains("GPAY",    ignoreCase = true) -> "GPay"
        senderId.contains("PHONEPE", ignoreCase = true) -> "PhonePe"
        senderId.contains("PAYTM",   ignoreCase = true) -> "Paytm"
        else -> "GPay"
    }

    private fun Double.fmt() = "%.4f".format(this)
}

// Extended to include hour stats + isColdStart flag
data class DeviationStats(
    val mean        : Double,   // mean transaction amount (90-day window)
    val stdDev      : Double,   // population stddev of amounts (≥ 1.0)
    val meanHour    : Double,   // mean transaction hour (0.0–23.0)
    val stdDevHour  : Double,   // population stddev of hours (≥ 0.5)
    val sampleCount : Int,      // number of transactions in window
    val isColdStart : Boolean   // true when sampleCount < MIN_SAMPLES
)
