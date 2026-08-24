package com.paysense.app.layer3

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

// Single source of the "paysense_prefs" SharedPreferences instance, backed by
// AndroidX Security's EncryptedSharedPreferences instead of a plain file --
// the JWT and is_authenticated flag are stored here (see FraudApiService,
// MainActivity, ProfileFragment), and a plain SharedPreferences file is
// readable in cleartext by anything with filesystem access to the app's data
// directory (root, a backup extraction, a forensic image). MODE_PRIVATE alone
// only stops OTHER APPS on a normal, non-rooted device -- this adds a second
// layer that survives those cases too. All three callers must go through this
// same instance: EncryptedSharedPreferences can't read values a plain
// SharedPreferences wrote, or vice versa.
private const val PREFS_NAME = "paysense_prefs"

object SecurePrefs {
    @Volatile private var instance: SharedPreferences? = null

    fun get(context: Context): SharedPreferences =
        instance ?: synchronized(this) {
            instance ?: build(context.applicationContext).also { instance = it }
        }

    private fun build(context: Context): SharedPreferences {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        return EncryptedSharedPreferences.create(
            context,
            PREFS_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    }
}
