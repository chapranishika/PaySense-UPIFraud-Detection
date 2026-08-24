# Android Client Security Review

**Update, 2026-08-24:** the original version of this document was manual
code review only, with no JDK available in this environment to build or
test anything. A portable Eclipse Temurin JDK 17 was installed for this
session afterward (no admin rights, extracted to a local directory) so all
five findings below marked FIXED could actually be compiled, assembled
into a real debug APK, and checked against the project's existing unit
tests — not just written and hoped correct. `./gradlew assembleDebug` and
`./gradlew testDebugUnitTest` both pass clean after every change.
(Finding #4's fix compiles and builds clean like the others, but its
one genuinely runtime-only property — the Android Keystore round-trip —
could not be live-verified in this environment; see its section below for
exactly why and what was actually attempted. Finding #5 and the two purely-
correctness bugs alongside it required a second, different JDK — Eclipse
Temurin's Windows build is missing the JPEG codec library Lint's icon
checker needs, so `./gradlew build`'s lint step never actually ran until
the Microsoft Build of OpenJDK was substituted in — see finding #5 for the
full account.)

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

### 4. JWT stored in plain (unencrypted) SharedPreferences — FIXED, compile/build verified; Keystore round-trip NOT live-verified

`FraudApiService.kt` and `MainActivity.kt` read/write the JWT via
`context.getSharedPreferences("paysense_prefs", MODE_PRIVATE)`.
`MODE_PRIVATE` restricts the file to the app's own UID under normal
sandboxing — not exploitable by another app on a non-rooted, non-debuggable
device — but it was still plaintext on disk.

**What changed:** a new `SecurePrefs.kt` centralizes all access to this
prefs file through `androidx.security:security-crypto`'s
`EncryptedSharedPreferences` (AES256_GCM master key, AES256_SIV key
encryption, AES256_GCM value encryption) instead of a plain
`getSharedPreferences()` call. All four call sites across
`FraudApiService.kt` (read in `AuthInterceptor`, write in `login()`, clear
in `clearAuth()`), `MainActivity.kt` (read `is_authenticated` on startup),
and `ProfileFragment.kt` (clear on logout) were updated together — this is
load-bearing, not cosmetic: `EncryptedSharedPreferences` can't read a value
a plain `SharedPreferences` wrote or vice versa, so a partial migration
would silently break auth state, not just leave it unencrypted.

**What was actually verified, and what wasn't:** `compileDebugKotlin`,
`assembleDebug` (full APK), `compileReleaseKotlin`, and `testDebugUnitTest`
all pass clean with the new dependency and code in place — the same bar as
findings #1–3. What could **not** be verified is the one thing that
actually matters for `EncryptedSharedPreferences`: that its first-run key
generation against the real Android Keystore succeeds and the value
round-trips correctly. That only happens at runtime on a device or
emulator, not in a compile check or a local-JVM unit test (this project's
existing unit tests already avoid the Android framework runtime
entirely — see `build.gradle.kts`'s `isReturnDefaultValues = true` comment).

**A real attempt was made to close that gap, not just asserted as
impossible.** This session set up a full local Android emulator from
scratch: downloaded and installed Android SDK cmdline-tools, accepted
licenses, installed an x86_64 API 34 `google_apis` system image, freed disk
space to fit it (see below), created an AVD, and got as far as the
emulator's own hypervisor-capability check reporting **WHPX available and
compatible**. It then failed silently at actual VM creation. Root cause,
confirmed directly: this session's process token is a member of
`BUILTIN\Administrators` but running **non-elevated**
(`IsInRole(Administrator)` returns `False` — classic UAC split-token
behavior), and while WHPX *capability detection* doesn't require elevation,
*creating* a hypervisor partition does. This is a permission this
non-interactive automated session cannot grant itself (no UAC prompt can be
answered here). Whoever runs this build interactively, in a normal elevated
or admin desktop session, would very likely not hit this and could complete
the live verification — the SDK/AVD setup is already done and left in
place for that (see below).

**Along the way:** the machine's C: drive was nearly full (1.3GB free, then
0 after a failed download), which independently blocked the system-image
install until `pip cache purge` and `npm cache clean --force` were run —
with explicit confirmation before either — freeing ~6.6GB. The AVD's data
partition also needed to be manually sized down (`-partition-size 2048` at
launch) to fit in the remaining space on E:. None of this is a reason the
encryption fix itself is wrong; it's exactly the kind of environment
friction a live device/CI run wouldn't have.

Lower severity than #1 even so, given the JWT is already short-lived
(~60 minutes) and there's no refresh-token to make persistent — but the
fix is real and shipped, not just documented as a good idea.

### 5. Two broadcast receivers were exported with no flag on pre-Android-13 — FIXED

Found by finally getting Android Lint to run to completion, not by manual
review (see below). `MainActivity.registerReceivers()` had a manual
`Build.VERSION_CODES.TIRAMISU` branch: on Android 13+ it correctly passed
`RECEIVER_NOT_EXPORTED`, but pre-13 it called the 2-arg `registerReceiver()`
with no flag at all — which defaults to **exported**. `categoryPromptReceiver`
and `fraudAlertReceiver` listen for `com.paysense.SHOW_CATEGORY_PROMPT` and
`com.paysense.FRAUD_ALERT_HIGH`, custom action strings with no permission
protection — meaning on any pre-Android-13 device, **any other installed
app** could broadcast either action and trigger these receivers directly,
spoofing a fake fraud alert or category prompt inside PaySense.

Fixed by replacing the manual branch with
`ContextCompat.registerReceiver(this, receiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)`
— the AndroidX compat call applies the not-exported flag correctly across
every API level in one call, closing the pre-13 exposure instead of just
satisfying newer devices.

**How this was found**: `./gradlew build` (the full lifecycle, lint
included) had never actually completed all session. The first portable
JDK used tonight (Eclipse Temurin) is missing `javajpeg.dll` — confirmed
by searching the entire JDK install for any JPEG codec library, genuinely
absent, not a config issue — which crashes Lint's `IconDetector` before
analysis can even start. Downloaded the Microsoft Build of OpenJDK 17
instead (built and tested specifically for Windows); confirmed
`javajpeg.dll` exists there, and Lint finally ran, surfacing this finding
plus two purely-correctness bugs (a `minSdk`-violating API call in the SMS
parser, a missing Play Store `<uses-feature>` declaration) — all three
fixed in the same pass. `./gradlew build` now completes with 0 lint
errors (was 8), 283 advisory warnings (dependency versions, hardcoded
strings — not correctness or security issues).

## Live-verifying the Keystore round-trip (left for an elevated session)

The Android SDK, system image, and AVD are already set up:
```
SDK root:  C:\Users\chapr\AppData\Local\Android\Sdk
           (cmdline-tools\latest, system-images on E: via a junction)
AVD home:  E:\android-sdk-data\avd  (AVD name: paysense_test)
```
From an **elevated** (Run as Administrator) terminal with `JAVA_HOME` set
to a JDK 17:
```
set ANDROID_AVD_HOME=E:\android-sdk-data\avd
set ANDROID_SDK_ROOT=C:\Users\chapr\AppData\Local\Android\Sdk
"%ANDROID_SDK_ROOT%\emulator\emulator.exe" -avd paysense_test -partition-size 2048
```
Then install and launch the app (`adb install -r app-debug.apk`), log in,
and confirm no crash and that a re-launch still shows the authenticated
state — the practical sign the Keystore round-trip actually works.

## Reproducing this check

```
cd PaySense-Android-Client-New
JAVA_HOME=<path to a JDK 17 install> ./gradlew build
```
Gradle wrapper is 9.1.0 (`gradle/wrapper/gradle-wrapper.properties`);
`app/build.gradle.kts` pins `sourceCompatibility`/`targetCompatibility` to
`JavaVersion.VERSION_17`, so a JDK 17 (not 21) install matches exactly.
**Use a JDK that actually ships JPEG codec support for AWT/ImageIO** —
Eclipse Temurin's Windows `jdk` archive does not (verified by searching the
whole install for any `*jpeg*` native library; genuinely absent), which
crashes `lintAnalyzeDebug` before analysis starts and silently prevents
`./gradlew build`'s lint step from ever running. The Microsoft Build of
OpenJDK 17 for Windows does ship it and was used to actually get Lint
running for finding #5 above.
