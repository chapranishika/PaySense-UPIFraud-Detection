# Android Client Security Review

**Update, 2026-08-24 → 2026-08-25:** the original version of this document
was manual code review only, with no JDK available in this environment to
build or test anything. A portable Eclipse Temurin JDK 17 was installed
afterward so all five findings below marked FIXED could actually be
compiled, assembled into a real debug APK, and checked against the
project's existing unit tests. Finding #5 needed a second, different
JDK — Eclipse Temurin's Windows build is missing the JPEG codec library
Lint's icon checker needs, so `./gradlew build`'s lint step never actually
ran until the Microsoft Build of OpenJDK was substituted in (see finding
#5 for the full account).

**Finding #4's Keystore round-trip — the one thing that stayed unverified
the longest — is now fully live-verified.** A real local Android emulator
was eventually built and run on this same machine (see finding #4's "How
the emulator blocker actually got resolved"), the app was installed, real
login was performed against a real running backend, the process was force-
stopped to clear all in-memory state, and a relaunch went straight to the
authenticated dashboard with no login prompt — direct proof the encrypted
storage actually round-trips through the real Android Keystore. Every
finding in this document is now either fixed-and-live-verified or
documented with the exact reasoning behind why it's out of scope.

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

### 4. JWT stored in plain (unencrypted) SharedPreferences — FIXED, fully live-verified

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

**Keystore round-trip: live-verified, on a real running emulator, 2026-08-25.**
Installed the app, logged in for real (`POST /auth/token`, 200, confirmed
in the backend's own request log — not just the app's UI), then
`am force-stop`'d the process entirely (clearing all in-memory state) and
relaunched. The app went straight to the authenticated dashboard with no
login prompt — proof the encryption key was generated against the real
Android Keystore on first write, persisted, and was successfully retrieved
and used to decrypt the stored JWT and `is_authenticated` flag on a
completely fresh process. This is the actual thing that mattered; compiling
cleanly was never proof of this specific behavior.

**How the emulator blocker actually got resolved** (kept here since it's
non-obvious and worth not rediscovering): the earlier theory — a
non-elevated process token blocking WHPX partition creation — turned out to
be incomplete. Testing with acceleration fully disabled (`-no-accel`)
produced the *exact same* failure, proving the real blocker was upstream of
WHPX entirely: a `STATUS_DLL_NOT_FOUND` failure to load
`api-ms-win-crt-utility-l1-1-0.dll`, confirmed via direct `LoadLibrary`
P/Invoke, native PowerShell process creation (ruling out a Bash/MSYS
quoting artifact), and a full PE import-table dependency walk. The DLL
itself was fine in isolation (`LoadLibrary` on it alone succeeded) — the
real issue was that invoking the qemu backend binary directly, bypassing
`emulator.exe`'s own `PATH`/library-search-path setup, meant its actual
runtime dependencies (`libandroid-emu-*.dll`, `libglib2*.dll`, several
`api-ms-win-crt-*.dll`), which live in `emulator/` and `emulator/lib64/`,
not next to the qemu binary, were never found. Fixing `PATH` to include
`emulator/`, `emulator/lib64/`, `emulator/lib64/gles_swiftshader/`, and
`emulator/lib64/vulkan/` before invoking qemu resolved it. A second,
separate issue then surfaced once past that: the Vulkan loader hardcodes
its ICD manifest search path relative to the qemu binary's own directory
(`emulator/qemu/windows-x86_64/lib/qemu/lib64/vulkan/`), which doesn't
exist — the real files live in `emulator/lib64/vulkan/`. Setting
`VK_ICD_FILENAMES` didn't help (the app overrides it internally regardless
of the environment), so the fix was creating the expected directory and
copying `vk_swiftshader_icd.json` + `vk_swiftshader.dll` there directly. A
last, trivial fix: `-partition-size 2048` is invalid (max is 2047), off by
one. With all three fixed and hardware acceleration re-enabled (WHPX was
never actually the blocker), the AVD booted fully in ~170 seconds.

**Along the way:** the machine's C: drive was nearly full (1.3GB free, then
0 after a failed download), which independently blocked the system-image
install until `pip cache purge` and `npm cache clean --force` were run —
with explicit confirmation before either — freeing ~6.6GB. None of this is
a reason the encryption fix itself is wrong; it's exactly the kind of
environment friction a live device/CI run wouldn't have.

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

## Running the emulator (done — this is the working recipe)

No elevation needed, in the end — see finding #4 above for why the earlier
"needs an elevated session" note was based on an incomplete diagnosis. The
Android SDK, system image, and AVD are already set up:
```
SDK root:  C:\Users\chapr\AppData\Local\Android\Sdk
           (cmdline-tools\latest, system-images on E: via a junction)
AVD home:  E:\android-sdk-data\avd  (AVD name: paysense_test)
```
The `emulator.exe` wrapper itself still doesn't surface the underlying
qemu-backend error clearly on this specific SDK download, so launch the
qemu backend directly with its dependency paths and Vulkan ICD location
set explicitly:
```powershell
$env:ANDROID_AVD_HOME = "E:\android-sdk-data\avd"
$env:ANDROID_SDK_ROOT = "C:\Users\chapr\AppData\Local\Android\Sdk"
$env:PATH = "$env:ANDROID_SDK_ROOT\emulator;$env:ANDROID_SDK_ROOT\emulator\lib64;$env:ANDROID_SDK_ROOT\emulator\lib64\gles_swiftshader;$env:ANDROID_SDK_ROOT\emulator\lib64\vulkan;$env:ANDROID_SDK_ROOT\emulator\qemu\windows-x86_64;$env:PATH"
& "$env:ANDROID_SDK_ROOT\emulator\qemu\windows-x86_64\qemu-system-x86_64-headless.exe" `
    -avd paysense_test -no-window -no-audio -no-boot-anim `
    -gpu swiftshader_indirect -no-snapshot -partition-size 2000
```
One-time setup the Vulkan loader needs (its ICD manifest path is hardcoded
relative to the qemu binary and doesn't match where the real files ship):
```powershell
$vk = "$env:ANDROID_SDK_ROOT\emulator\qemu\windows-x86_64\lib\qemu\lib64\vulkan"
New-Item -ItemType Directory -Force -Path $vk
Copy-Item "$env:ANDROID_SDK_ROOT\emulator\lib64\vulkan\vk_swiftshader_icd.json" $vk
Copy-Item "$env:ANDROID_SDK_ROOT\emulator\lib64\vulkan\vk_swiftshader.dll" $vk
```
Then install and launch the app:
```
adb install -r app-debug.apk
adb shell am start -n com.paysense.app/.ui.MainActivity
```
To verify the Keystore round-trip specifically: log in, then
`adb shell am force-stop com.paysense.app` followed by the `am start`
above again — a re-launch straight to the dashboard with no login prompt
confirms it. This was done in this session on 2026-08-25; see finding #4.

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
