# Android Client Security Review

**Update, 2026-08-24:** the original version of this document was manual
code review only, with no JDK available in this environment to build or
test anything. A portable Eclipse Temurin JDK 17 was installed for this
session afterward (no admin rights, extracted to a local directory) so the
three findings below marked FIXED could actually be compiled, assembled
into a real debug APK, and checked against the project's existing unit
tests — not just written and hoped correct. `./gradlew assembleDebug` and
`./gradlew testDebugUnitTest` both pass clean after all three changes.

Scope: `PaySense-Android-Client-New/app/src/main/kotlin/com/paysense/app/`,
focused on the login flow, token handling, and network layer.

## What was checked and found clean

- **Login flow has no client-side credential comparison.** `MainActivity`'s
  login handler calls `FraudApiService.login()`, which makes a real
  `POST /auth/token` call and trusts only the server's response — confirmed
  by reading `FraudApiService.kt:127-151`. This was the "auth theater" bug
  found and fixed earlier tonight; re-reading it now confirms the fix holds.
- **Transport is HTTPS.** `BASE_URL = "https://paysense-api.onrender.com/"`
  (`FraudApiService.kt:82`). No endpoint in `PaySenseApi` uses `http://`.

## Fixed and verified via a real build

### 1. Full request/response bodies, including the JWT, were logged unconditionally — FIXED

`FraudApiService.kt`: `HttpLoggingInterceptor`'s `Level.BODY` was
unconditional — running in release builds too, logging the
`Authorization: Bearer <jwt>` header `AuthInterceptor` attaches to every
call and every transaction payload sent to `/predict` (amounts, VPAs,
device info) straight to Logcat. This was the one finding with a plausible
real-world exploitation path on a stock device. Now gated on
`BuildConfig.DEBUG`:
```kotlin
val logging = HttpLoggingInterceptor { Log.d(TAG, "HTTP | $it") }
    .apply { level = if (BuildConfig.DEBUG) HttpLoggingInterceptor.Level.BODY
                      else HttpLoggingInterceptor.Level.NONE }
```
Verified, not assumed: `compileReleaseKotlin` succeeded, and the generated
`app/build/generated/source/buildConfig/release/.../BuildConfig.java`
was read directly to confirm `DEBUG = false` in that variant (`debug`'s
generated file confirms `DEBUG = true` there) — so the gate resolves the
way it's meant to in both variants, not just in the source code's intent.

### 2. Logout cleared `is_authenticated` but not the stored token — FIXED

`ProfileFragment.kt`'s logout handler only flipped `is_authenticated` to
`false`, unlike `FraudApiService.clearAuth()` (used on a 401), which
removes the token too. A still-valid JWT (up to ~60 min server-side) sat in
SharedPreferences after a user believed they'd logged out. Now removes
`auth_token` as well, matching `clearAuth()`'s behavior.

### 3. `android:usesCleartextTraffic="true"` in the manifest — FIXED

Not exploited (every endpoint is already `https://`), but an unnecessary
standing permission. Flipped to `false` in `AndroidManifest.xml`; the debug
APK still assembles clean (`processDebugMainManifest` succeeded), so
nothing in the app was relying on cleartext traffic being allowed.

## Documented, not fixed

### 4. JWT stored in plain (unencrypted) SharedPreferences

`FraudApiService.kt` and `MainActivity.kt` read/write the JWT via
`context.getSharedPreferences("paysense_prefs", MODE_PRIVATE)`.
`MODE_PRIVATE` restricts the file to the app's own UID under normal
sandboxing — not exploitable by another app on a non-rooted, non-debuggable
device — but it's still plaintext on disk. The standard hardening
(Jetpack Security's `EncryptedSharedPreferences`) was deliberately not
applied here: it requires adding a new dependency
(`androidx.security:security-crypto`) and, more importantly, its
first-run key generation goes through the Android Keystore — something a
compile check can't verify, since it only exists at runtime on a real
device or emulator, neither of which is available in this environment.
Getting the *code* to compile isn't the same bar as getting encrypted
storage to actually round-trip correctly on first launch, so this was left
as a documented finding rather than a change nobody could verify beyond
"it compiles." Lower severity than #1 given the JWT is already short-lived
(~60 minutes) and there's no refresh-token to make persistent.

## Reproducing this check

```
cd PaySense-Android-Client-New
JAVA_HOME=<path to a JDK 17 install> ./gradlew assembleDebug compileReleaseKotlin testDebugUnitTest
```
Gradle wrapper is 9.1.0 (`gradle/wrapper/gradle-wrapper.properties`);
`app/build.gradle.kts` pins `sourceCompatibility`/`targetCompatibility` to
`JavaVersion.VERSION_17`, so a JDK 17 (not 21) install matches exactly.
