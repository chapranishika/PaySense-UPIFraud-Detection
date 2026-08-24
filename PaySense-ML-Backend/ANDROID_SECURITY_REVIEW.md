# Android Client Security Review

Manual code review only. `./gradlew --version` fails in this environment
("JAVA_HOME is not set and no 'java' command could be found in your PATH")
— there is no JDK available, so nothing below was built, run, or tested.
Findings are from reading the Kotlin source directly; no code changes were
made to the Android client for this reason. This is a deliberate choice,
not an oversight: writing Kotlin changes that can't be verified to compile
would violate the standard the rest of tonight's fixes were held to.

Scope: `PaySense-Android-Client-New/app/src/main/kotlin/com/paysense/app/`,
focused on the login flow, token handling, and network layer (the areas
most likely to carry real security defects, versus e.g. UI/layout code).

## What was checked and found clean

- **Login flow has no client-side credential comparison.** `MainActivity`'s
  login handler calls `FraudApiService.login()`, which makes a real
  `POST /auth/token` call and trusts only the server's response — confirmed
  by reading `FraudApiService.kt:127-151`. This was the "auth theater" bug
  found and fixed earlier tonight; re-reading it now confirms the fix holds
  and there's no second, dormant client-side check elsewhere.
- **Transport is HTTPS.** `BASE_URL = "https://paysense-api.onrender.com/"`
  (`FraudApiService.kt:82`). No endpoint in `PaySenseApi` uses `http://`.

## Findings (not fixed — see note above)

### 1. Full request/response bodies, including the JWT, are logged unconditionally

`FraudApiService.kt:96-97`:
```kotlin
val logging = HttpLoggingInterceptor { Log.d(TAG, "HTTP | $it") }
    .apply { level = HttpLoggingInterceptor.Level.BODY }
```
This interceptor is not gated behind `BuildConfig.DEBUG` or any build-variant
check, so it runs in release builds too. `Level.BODY` logs full request and
response bodies to Logcat — including the `Authorization: Bearer <jwt>`
header `AuthInterceptor` attaches to every call, and every transaction
payload (amounts, VPAs, device info) sent to `/predict`. On a device where
another app holds `READ_LOGS` (pre-Android 4.1 behavior, or any rooted /
debuggable device), this is a real credential- and PII-leak surface. The fix
is straightforward — gate the interceptor on `BuildConfig.DEBUG`, or drop
the level to `Level.BASIC`/`NONE` in release — but is left undone here since
it can't be compiled and verified in this environment.

### 2. JWT stored in plain (unencrypted) SharedPreferences

`FraudApiService.kt:41-42, 139-143` and `MainActivity.kt:119` all read/write
the JWT via `context.getSharedPreferences("paysense_prefs", MODE_PRIVATE)`.
`MODE_PRIVATE` restricts the file to the app's own UID under normal
Android sandboxing, so this is not exploitable by another app on a
non-rooted, non-debuggable device — but it is still plaintext on disk, and
the standard hardening (Jetpack Security's `EncryptedSharedPreferences`) is
not in use. Lower severity than #1 given the JWT is already short-lived
(`expires_in` ~60 minutes server-side, confirmed in `ApiModels.kt:208-209`)
and there's no refresh-token to make persistent.

### 3. Logout clears `is_authenticated` but not the stored token

`ProfileFragment.kt:51-54`:
```kotlin
binding.btnProfileLogout.setOnClickListener {
    val prefs = requireContext().getSharedPreferences("paysense_prefs", Context.MODE_PRIVATE)
    prefs.edit().putBoolean("is_authenticated", false).apply()
    (activity as? MainActivity)?.showLoginOverlay()
}
```
This only flips `is_authenticated` to `false`; it never removes
`auth_token`, unlike `FraudApiService.clearAuth()` (used on a 401 response),
which does both. The stale token sits in SharedPreferences until it expires
or a 401 triggers `clearAuth()`. Since the login overlay blocks further
UI-driven calls once `is_authenticated=false`, this isn't directly
exploitable through the app itself — but it's an inconsistency between two
places that should do the same thing, and the stale token is a live JWT for
up to an hour after a user believes they've logged out.

### 4. `android:usesCleartextTraffic="true"` in the manifest

`AndroidManifest.xml:18`. Not currently exploited — `BASE_URL` is `https://`
and no code path constructs an `http://` URL — but it's a standing
permission the app doesn't need, and it would silently allow a future `http`
URL (typo, a debug endpoint left in, a malicious library) to work instead of
failing loudly. Setting this to `false` (or scoping it via a
`networkSecurityConfig`) costs nothing given HTTPS is already the only
transport in use.

## Severity ordering

\#1 (unconditional Logcat body logging) is the one worth prioritizing if this
ever gets picked up with real Android tooling — it's the only one of the
four with a plausible real-world exploitation path on a stock device
(anything that can read this app's logcat output, including other apps on
older Android versions, or a device the user has rooted/debugged
themselves). \#2–4 all require either physical/root access or a scenario
that's already out of scope for `MODE_PRIVATE`'s threat model.
