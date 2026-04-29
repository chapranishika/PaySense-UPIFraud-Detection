package com.paysense.app.layer2

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

// ============================================================================
//  PaySenseDatabase.kt  (v2 — adds TransactionHistory entity and DAO)
// ============================================================================
@Database(
    entities  = [PayeeCache::class, TransactionHistory::class],
    version   = 2,             // Bumped from 1 → new entity added
    exportSchema = true
)
abstract class PaySenseDatabase : RoomDatabase() {

    abstract fun payeeDao(): PayeeDao
    abstract fun transactionDao(): TransactionDao

    companion object {
        @Volatile private var INSTANCE: PaySenseDatabase? = null

        fun getInstance(context: Context): PaySenseDatabase =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    PaySenseDatabase::class.java,
                    "paysense_local.db"
                )
                .fallbackToDestructiveMigration()   // Dev only
                .build()
                .also { INSTANCE = it }
            }
    }
}
