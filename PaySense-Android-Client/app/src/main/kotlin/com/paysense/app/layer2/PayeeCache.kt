package com.paysense.app.layer2

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * ============================================================================
 *  PaySense — PayeeCache.kt
 *  Layer 2: Room Database Entity — Local Payee-to-Category mapping table
 *
 *  This table is the persistent, on-device Human-in-the-Loop cache.
 *  It answers one question: "Has the user ever categorised this payee before?"
 *
 *  Schema:
 *    payeeName  → PRIMARY KEY. Normalised to lowercase for case-insensitive
 *                 matching ("Zomato", "zomato", "ZOMATO" → same row).
 *    category   → User-selected or NLP-inferred category string.
 *                 Examples: "Food", "Travel", "Uncategorized"
 *    source     → Who assigned this category:
 *                   "user"  → manually selected in the HITL bottom sheet
 *                   "nlp"   → assigned by FinText-6K classifier (conf ≥ 0.65)
 *                   "auto"  → assigned by keyword rule (e.g., "ZOMATO" → Food)
 *    confidence → NLP confidence at time of assignment (null if source="user")
 *    updatedAt  → Epoch ms of last update. Allows stale-cache invalidation
 *                 (e.g., re-prompt if source="nlp" and updatedAt > 90 days).
 *
 *  Room auto-generates the CREATE TABLE SQL from these annotations.
 * ============================================================================
 */
@Entity(tableName = "payee_category_cache")
data class PayeeCache(

    @PrimaryKey
    val payeeName   : String,   // Normalised to lowercase on insert

    val category    : String,   // "Food", "Travel", "Shopping", etc.

    val source      : String,   // "user" | "nlp" | "auto"

    val confidence  : Float?,   // NLP confidence score, null for user/auto

    val updatedAt   : Long = System.currentTimeMillis()
)
