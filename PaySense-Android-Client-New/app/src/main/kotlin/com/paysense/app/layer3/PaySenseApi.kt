package com.paysense.app.layer3

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

/**
 * ============================================================================
 *  PaySense — PaySenseApi.kt
 *  Layer 3: Retrofit Interface — HTTP contract for the FastAPI backend
 *
 *  Retrofit reads this interface at runtime and generates a concrete
 *  implementation that handles JSON serialisation, HTTP execution,
 *  and coroutine suspension — you write zero networking code.
 * ============================================================================
 */
interface PaySenseApi {

    /**
     * POST /auth/token
     * Exchanges the demo username/password for a genuine JWT access token —
     * the credential check happens server-side (main.py's TokenRequest /
     * get_token), not in this client. Public endpoint, no auth header
     * required to call it. Returns 401 if credentials are wrong.
     */
    @POST("/auth/token")
    suspend fun login(
        @Body request: TokenRequest
    ): Response<TokenResponse>

    /**
     * POST /predict
     * Sends a [TransactionRequest] JSON body to the FastAPI backend and
     * returns a [TransactionResponse] containing the fraud verdict.
     *
     * Using Response<T> wrapper (instead of just T) gives us access to the
     * HTTP status code, which lets us distinguish between:
     *   200 OK      → successful prediction
     *   422 Unprocessable Entity → Pydantic validation failed (bad input)
     *   500         → server-side inference error
     */
    @POST("/predict")
    suspend fun predictFraud(
        @Body request: TransactionRequest
    ): Response<TransactionResponse>

    /**
     * GET /health
     * Lightweight ping to confirm the server is up before sending a
     * prediction request. Call this once on app launch, not on every SMS.
     */
    @GET("/health")
    suspend fun healthCheck(): Response<Map<String, Any>>

    /**
     * POST /classify
     * Layer 2, Tier 2 of the payee-category resolution pipeline: sends the
     * raw SMS narration body to the server-side NLP classifier (TF-IDF +
     * calibrated LinearSVC, trained on FinText-6K) and gets back a category
     * + confidence. Only called by PayeeCacheRepository on a Tier-1 cache
     * miss — see resolveCategory() in PayeeCacheRepository.kt.
     *
     * The model was trained on exactly five classes — Food, Travel, EMI,
     * Investment, Shopping — and will never return anything outside that
     * set. The caller applies NLP_CONFIDENCE_THRESHOLD (0.65) before
     * trusting the result; below that (or on any failure/exception) it
     * falls through to the Tier-3 human-in-the-loop prompt.
     */
    @POST("/classify")
    suspend fun classifyCategory(
        @Body request: CategoryRequest
    ): Response<CategoryResponse>

    /**
     * GET /insights/weekly
     * Retrieve weekly spending savings tips from FastAPI backend.
     */
    @GET("/insights/weekly")
    suspend fun getWeeklyInsights(
        @retrofit2.http.Query("total_spent") totalSpent: Double,
        @retrofit2.http.Query("top_category") topCategory: String,
        @retrofit2.http.Query("top_category_pct") topCategoryPct: Double,
        @retrofit2.http.Query("fraud_alerts") fraudAlerts: Int,
        @retrofit2.http.Query("vs_last_week_pct") vsLastWeekPct: Double = 12.0
    ): Response<WeeklyInsight>
}
