package com.paysense.app.layer2

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class PayeeCacheRepository private constructor(private val context: Context) {

    private val dao: PayeeDao = PaySenseDatabase.getInstance(context).payeeDao()

    companion object {
        @Volatile private var INSTANCE: PayeeCacheRepository? = null
        fun getInstance(context: Context): PayeeCacheRepository =
            INSTANCE ?: synchronized(this) { INSTANCE ?: PayeeCacheRepository(context).also { INSTANCE = it } }
    }

    suspend fun getCategoryForPayee(rawPayee: String): String? = withContext(Dispatchers.IO) {
        val normalised = rawPayee.trim().lowercase()
        val cached = dao.getCategoryForPayee(normalised)
        cached?.category
    }

    suspend fun saveUserCategory(rawPayee: String, category: String) = withContext(Dispatchers.IO) {
        val normalised = rawPayee.trim().lowercase()
        val entry = PayeeCache(payeeName = normalised, category = category, source = "user", confidence = null)
        dao.insertPayeeCategory(entry)
    }
}
