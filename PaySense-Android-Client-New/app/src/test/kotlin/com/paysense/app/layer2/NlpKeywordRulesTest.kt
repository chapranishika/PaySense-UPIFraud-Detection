package com.paysense.app.layer2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for [NlpKeywordRules], the Tier-2 keyword-shortcut classifier
 * that [PayeeCacheRepository.resolveCategory] consults before falling back
 * to Tier-3 HITL. The table/matching logic was extracted out of the
 * repository class (which requires a Context + Room DB to construct) into
 * this small standalone object so it can be tested directly.
 */
class NlpKeywordRulesTest {

    @Test
    fun `known food merchants resolve to Food at 0point99 confidence`() {
        assertEquals("Food" to 0.99f, NlpKeywordRules.classify("ZOMATO"))
        assertEquals("Food" to 0.99f, NlpKeywordRules.classify("Swiggy"))
        assertEquals("Food" to 0.99f, NlpKeywordRules.classify("Dominos Pizza"))
    }

    @Test
    fun `irctc resolves to Travel at 0point99 confidence`() {
        val result = NlpKeywordRules.classify("IRCTC")
        assertEquals("Travel", result?.first)
        assertEquals(0.99f, result?.second)
    }

    @Test
    fun `ola and uber resolve to Travel at 0point97 confidence`() {
        assertEquals("Travel" to 0.97f, NlpKeywordRules.classify("OLA"))
        assertEquals("Travel" to 0.97f, NlpKeywordRules.classify("Uber India"))
    }

    @Test
    fun `amazon and flipkart resolve to Shopping at 0point98 confidence`() {
        assertEquals("Shopping" to 0.98f, NlpKeywordRules.classify("AMAZON PAY"))
        assertEquals("Shopping" to 0.98f, NlpKeywordRules.classify("Flipkart Internet"))
    }

    @Test
    fun `netflix and hotstar resolve to Entertainment at 0point98 confidence`() {
        assertEquals("Entertainment" to 0.98f, NlpKeywordRules.classify("NETFLIX.COM"))
        assertEquals("Entertainment" to 0.98f, NlpKeywordRules.classify("Disney Hotstar"))
    }

    @Test
    fun `telecom recharges resolve to Recharge at 0point95 confidence`() {
        assertEquals("Recharge" to 0.95f, NlpKeywordRules.classify("AIRTEL"))
        assertEquals("Recharge" to 0.95f, NlpKeywordRules.classify("JIO Prepaid"))
        assertEquals("Recharge" to 0.95f, NlpKeywordRules.classify("BSNL"))
    }

    @Test
    fun `all keyword rule confidences meet the high-confidence bar of 0point95 or above`() {
        // The whole point of the keyword shortcut table is that it only
        // fires for globally unambiguous merchants — every entry must clear
        // the auto-assign threshold (0.65) by a wide margin.
        for ((keyword, rule) in NlpKeywordRules.rules) {
            assertTrue(
                "confidence for '$keyword' should be >= 0.95f but was ${rule.second}",
                rule.second >= 0.95f
            )
        }
    }

    @Test
    fun `matching is case-insensitive and substring-based`() {
        assertEquals("Food" to 0.99f, NlpKeywordRules.classify("ZOMATO ORDER #4521"))
        assertEquals("Food" to 0.99f, NlpKeywordRules.classify("zomato online food ordering"))
    }

    @Test
    fun `unknown payee falls through to null and triggers HITL`() {
        assertNull(NlpKeywordRules.classify("Rajesh Kumar"))
        assertNull(NlpKeywordRules.classify("XYZ CORP PVT LTD"))
        assertNull(NlpKeywordRules.classify(""))
    }
}
