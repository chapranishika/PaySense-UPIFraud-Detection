package com.paysense.app.layer3

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for [DeviationStatsCalculator] — the population mean/stddev
 * z-score math behind PaySense's "personalized anomaly scoring" claim.
 * This is a pure function extracted out of
 * FraudApiService.computeDeviationStats() (which otherwise needs a Context
 * + Room DB to run) so the statistics themselves can be verified directly,
 * independent of any database or network access.
 *
 * Expected mean/stdDev/z-score values below were independently
 * cross-checked with a population-statistics reference implementation
 * before being hard-coded as expectations here.
 */
class DeviationStatsCalculatorTest {

    private val delta = 0.0001

    @Test
    fun `cold start with fewer than MIN_SAMPLES transactions returns neutral zero-signal stats`() {
        val amounts = listOf(100.0, 200.0, 150.0) // only 3, MIN_SAMPLES = 5
        val hours = listOf(9, 10, 11)

        val stats = DeviationStatsCalculator.compute(amounts, hours)

        assertTrue(stats.isColdStart)
        assertEquals(0, stats.sampleCount)
        assertEquals(0.0, stats.mean, delta)
        assertEquals(1.0, stats.stdDev, delta)
        assertEquals(12.0, stats.meanHour, delta)
        assertEquals(6.0, stats.stdDevHour, delta)
    }

    @Test
    fun `exactly MIN_SAMPLES transactions is enough to leave cold start`() {
        val amounts = listOf(100.0, 200.0, 300.0, 400.0, 500.0) // exactly 5
        val hours = listOf(10, 11, 12, 13, 14)

        val stats = DeviationStatsCalculator.compute(amounts, hours)

        assertFalse(stats.isColdStart)
        assertEquals(5, stats.sampleCount)
    }

    @Test
    fun `tight-spending user getting a large outlier transaction produces a large z-score`() {
        // Population mean = 500, population stdDev ~= 70.71
        val amounts = listOf(400.0, 450.0, 500.0, 550.0, 600.0)
        val hours = listOf(9, 10, 11, 12, 13)

        val stats = DeviationStatsCalculator.compute(amounts, hours)

        assertFalse(stats.isColdStart)
        assertEquals(500.0, stats.mean, delta)
        assertEquals(70.71067811865476, stats.stdDev, delta)

        val incomingAmount = 15000.0
        val zAmount = (incomingAmount - stats.mean) / stats.stdDev

        assertEquals(205.06096654409876, zAmount, delta)
        assertTrue("expected a large z-score for a tight spender, got $zAmount", zAmount > 50.0)
    }

    @Test
    fun `same outlier transaction for a high-variance user produces a small z-score`() {
        // Population mean = 12000, population stdDev ~= 2828.43
        val amounts = listOf(8000.0, 10000.0, 12000.0, 14000.0, 16000.0)
        val hours = listOf(9, 10, 11, 12, 13)

        val stats = DeviationStatsCalculator.compute(amounts, hours)

        assertFalse(stats.isColdStart)
        assertEquals(12000.0, stats.mean, delta)
        assertEquals(2828.42712474619, stats.stdDev, delta)

        val incomingAmount = 15000.0
        val zAmount = (incomingAmount - stats.mean) / stats.stdDev

        assertEquals(1.0606601717798212, zAmount, delta)
        assertTrue("expected a small z-score for a high-variance user, got $zAmount", zAmount < 2.0)
    }

    @Test
    fun `identical historical amounts clamp stdDev to 1point0 instead of dividing by zero`() {
        // Population variance of identical values is 0.0 -> sqrt(0) = 0.0,
        // which must be clamped to >= 1.0 so amountDeviationScore never
        // divides by zero (NaN/Infinity would corrupt the fraud model input).
        val amounts = listOf(500.0, 500.0, 500.0, 500.0, 500.0)
        val hours = listOf(9, 10, 11, 12, 13)

        val stats = DeviationStatsCalculator.compute(amounts, hours)

        assertFalse(stats.isColdStart)
        assertEquals(500.0, stats.mean, delta)
        assertEquals(1.0, stats.stdDev, delta) // clamped, not 0.0

        val zAmount = (510.0 - stats.mean) / stats.stdDev
        assertEquals(10.0, zAmount, delta)
        assertTrue(zAmount.isFinite())
    }

    @Test
    fun `identical historical hours clamp stdDevHour to 0point5 instead of dividing by zero`() {
        val amounts = listOf(100.0, 200.0, 300.0, 400.0, 500.0)
        val hours = listOf(10, 10, 10, 10, 10)

        val stats = DeviationStatsCalculator.compute(amounts, hours)

        assertFalse(stats.isColdStart)
        assertEquals(10.0, stats.meanHour, delta)
        assertEquals(0.5, stats.stdDevHour, delta) // clamped, not 0.0
    }

    @Test
    fun `fewer than MIN_SAMPLES hour rows fall back to neutral hour stats while amount stats stay real`() {
        // Enough amount history, but hourOfDay is null on most legacy rows
        // (pre-migration) so getHoursSince() only returns 2 rows.
        val amounts = listOf(400.0, 450.0, 500.0, 550.0, 600.0)
        val hours = listOf(9, 10) // fewer than MIN_SAMPLES

        val stats = DeviationStatsCalculator.compute(amounts, hours)

        assertFalse(stats.isColdStart)
        // Amount stats are computed normally...
        assertEquals(500.0, stats.mean, delta)
        assertEquals(70.71067811865476, stats.stdDev, delta)
        // ...but the hour component falls back to neutral values.
        assertEquals(12.0, stats.meanHour, delta)
        assertEquals(6.0, stats.stdDevHour, delta)
    }
}
