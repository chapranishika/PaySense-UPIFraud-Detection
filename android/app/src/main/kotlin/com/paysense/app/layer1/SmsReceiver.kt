package com.paysense.app.layer1

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.util.Log

class SmsReceiver : BroadcastReceiver() {
    companion object {
        private const val TAG = "PaySense_Demo_SMS"
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) return

        val messages = try {
            Telephony.Sms.Intents.getMessagesFromIntent(intent)
        } catch (e: Exception) {
            null
        }

        if (!messages.isNullOrEmpty()) {
            Log.d(TAG, "📩 SMS Received: ${messages[0].messageBody}")
            // NOTE: In DEMO mode, we do not process SMS to avoid any possible ANR.
            // Use the "Test Fraud Detection" button in the app for the demo pipeline.
        }
    }
}
