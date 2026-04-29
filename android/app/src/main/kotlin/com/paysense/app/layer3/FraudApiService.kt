package com.paysense.app.layer3

import android.content.Context
import com.paysense.app.layer2.PaySenseDatabase
import com.paysense.app.layer2.TransactionDao

class FraudApiService private constructor(context: Context) {

    companion object {
        @Volatile private var INSTANCE: FraudApiService? = null
        fun getInstance(context: Context): FraudApiService =
            INSTANCE ?: synchronized(this) { INSTANCE ?: FraudApiService(context.applicationContext).also { INSTANCE = it } }
    }

    private val txnDao: TransactionDao = PaySenseDatabase.getInstance(context).transactionDao()

    // NOTE: In DEMO mode, fraud scoring logic is simulated in MainActivity.
    // Real API calls and background processing are disabled to ensure zero ANR risk.
}
