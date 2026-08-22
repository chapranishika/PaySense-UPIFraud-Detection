package com.paysense.app.layer1

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for the three deterministic SMS gates in [SmsReceiver]:
 *   Gate 1 — TRAI sender ID format regex
 *   Gate 2 — transaction keyword regex
 *   Gate 3 — named-group extraction regex (+ quarantine on unparseable amount)
 *
 * [SmsReceiver] extends android.content.BroadcastReceiver but none of the
 * methods under test touch anything beyond android.util.Log, which the
 * module's `testOptions.unitTests.isReturnDefaultValues = true` setting
 * (app/build.gradle.kts) makes safe to call from a plain local JVM test —
 * no Robolectric/instrumentation required.
 */
class SmsReceiverTest {

    private lateinit var receiver: SmsReceiver

    @Before
    fun setUp() {
        receiver = SmsReceiver()
    }

    // ── Gate 1: TRAI sender ID format ───────────────────────────────────────

    @Test
    fun `gate1 accepts standard 2-letter prefix TRAI sender ids`() {
        assertTrue(receiver.passesGate1("HD-HDFCBK"))
        assertTrue(receiver.passesGate1("VM-SBIINB"))
        assertTrue(receiver.passesGate1("AX-ICICIB"))
    }

    @Test
    fun `gate1 is case insensitive`() {
        assertTrue(receiver.passesGate1("hd-hdfcbk"))
    }

    @Test
    fun `gate1 rejects a personal 10-digit phone number sender`() {
        assertFalse(receiver.passesGate1("9876543210"))
    }

    @Test
    fun `gate1 rejects sender ids missing the hyphen`() {
        assertFalse(receiver.passesGate1("HDHDFCBK"))
    }

    @Test
    fun `gate1 rejects a suffix shorter than 4 characters`() {
        assertFalse(receiver.passesGate1("VM-AB"))
    }

    @Test
    fun `gate1 rejects a suffix longer than 6 characters`() {
        assertFalse(receiver.passesGate1("VM-HDFCBANK"))
    }

    // ── Gate 2: transaction keyword presence ────────────────────────────────

    @Test
    fun `gate2 passes on a real debit alert`() {
        assertTrue(
            receiver.passesGate2(
                "Rs.15000.00 debited from A/c XX1234 to AMAZON PAY for order UPI Ref 123456789012 on 22-Aug-2024"
            )
        )
    }

    @Test
    fun `gate2 passes for each supported keyword`() {
        assertTrue(receiver.passesGate2("Your account has been credited with Rs.2000"))
        assertTrue(receiver.passesGate2("UPI transaction of Rs.100 completed"))
        assertTrue(receiver.passesGate2("Payment of INR 500 received"))
        assertTrue(receiver.passesGate2("A transaction was made on your card"))
    }

    @Test
    fun `gate2 fails a promotional bank SMS from a valid TRAI sender`() {
        // Passes Gate 1 (valid TRAI-format sender)...
        assertTrue(receiver.passesGate1("HD-HDFCBK"))
        // ...but the body has no transactional keyword, so Gate 2 must reject it.
        val promoBody =
            "Congratulations! You have won a lucky draw prize. " +
                "Click here to claim your reward now: bit.ly/xyz123"
        assertFalse(receiver.passesGate2(promoBody))
    }

    // ── Gate 3: named-group extraction ──────────────────────────────────────

    @Test
    fun `gate3 extracts amount payee txnId and date from a real bank SMS`() {
        val body =
            "Rs.15000.00 debited from A/c XX1234 to AMAZON PAY for order UPI Ref 123456789012 on 22-Aug-2024"

        val parsed = receiver.applyGate3(sender = "HD-HDFCBK", body = body, timestamp = 1_724_000_000_000L)

        assertNotNull(parsed)
        parsed!!
        assertEquals(15000.00, parsed.amount, 0.0001)
        assertEquals("AMAZON PAY", parsed.payee)
        assertEquals("123456789012", parsed.txnId)
        assertEquals("22-Aug-2024", parsed.date)
        assertEquals("HD-HDFCBK", parsed.senderId)
    }

    @Test
    fun `gate3 quarantines a transactional SMS with no parseable amount`() {
        // Contains a Gate-2 keyword ("debited") and a payee/txnId/date, but
        // the amount is nowhere in a Rs./INR/₹-prefixed numeric form, so the
        // mandatory 'amount' capture group never fires.
        val body =
            "INR debited from your account towards ELECTRICITY BOARD for bill UPI Ref 987654321098 on 01-09-2024"

        assertTrue(receiver.passesGate2(body))

        val parsed = receiver.applyGate3(sender = "HD-HDFCBK", body = body, timestamp = 1_724_000_000_000L)

        assertNull("Gate 3 must quarantine (return null) when amount cannot be extracted", parsed)
    }

    @Test
    fun `gate3 falls back to synthetic txnId and empty date when absent`() {
        val body = "Rs.250 debited to CHAI POINT for snack"

        val parsed = receiver.applyGate3(sender = "HD-HDFCBK", body = body, timestamp = 42L)

        assertNotNull(parsed)
        parsed!!
        assertEquals(250.0, parsed.amount, 0.0001)
        assertEquals("TXN_42", parsed.txnId)
        assertEquals("", parsed.date)
    }
}
