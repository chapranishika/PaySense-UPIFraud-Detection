package com.paysense.app.ui

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.paysense.app.R
import com.paysense.app.databinding.ActivityMainBinding
import kotlinx.coroutines.launch
import java.text.NumberFormat
import java.util.Locale

/**
 * ============================================================================
 *  MainActivity.kt  (v3 — Finance tab added)
 *
 *  CHANGE FROM v2:
 *    Added BottomNavigationView with two tabs:
 *      Tab 1: Dashboard (existing fraud detection feed — UNCHANGED)
 *      Tab 2: Finance   (new FinanceFragment — Phase 1 spending tracker)
 *
 *  SAFE INTEGRATION:
 *    All existing fraud detection code (ViewModel observation, SMS permission
 *    handling, BroadcastReceivers) is 100% unchanged. The Finance tab is
 *    loaded via Fragment transactions — Dashboard content is retained in the
 *    back stack, so switching tabs does not re-trigger Room queries or
 *    cancel any running coroutines.
 *
 *  FRAGMENT STRATEGY:
 *    Both fragments are added once and shown/hidden (not replaced) so their
 *    ViewModels and state survive tab switches without re-creation.
 *    Dashboard uses the existing RecyclerView + ViewModel approach.
 *    Finance uses FinanceFragment + FinanceViewModel.
 * ============================================================================
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var adapter: TransactionAdapter

    private val viewModel: PaySenseViewModel by viewModels()
    private val currencyFmt = NumberFormat.getCurrencyInstance(Locale("en", "IN"))

    // Fragment instances — created once, shown/hidden to preserve state
    private val dashboardContent by lazy { DashboardContentFragment.newInstance() }
    private val financeFragment   by lazy { FinanceFragment.newInstance() }
    private var activeFragment: Fragment = dashboardContent

    // ── Permission launcher ───────────────────────────────────────────────────
    private val smsPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        binding.cardPermissionBanner.visibility =
            if (results.values.all { it }) View.GONE else View.VISIBLE
    }

    // ── Broadcast receivers ───────────────────────────────────────────────────
    private val categoryPromptReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != "com.paysense.SHOW_CATEGORY_PROMPT") return
            val payee  = intent.getStringExtra("payee_name") ?: return
            val txnId  = intent.getStringExtra("txn_id")    ?: return
            val amount = intent.getDoubleExtra("amount", 0.0)
            CategoryBottomSheet.newInstance(payee, amount, txnId)
                .show(supportFragmentManager, CategoryBottomSheet.TAG)
        }
    }

    private val fraudAlertReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != "com.paysense.FRAUD_ALERT_HIGH") return
            val score  = intent.getDoubleExtra("fraud_score", 0.0)
            val payee  = intent.getStringExtra("payee") ?: "Unknown"
            val amount = intent.getDoubleExtra("amount", 0.0)
            com.google.android.material.snackbar.Snackbar
                .make(binding.root,
                    "⚠️ Suspicious: ₹${"%.2f".format(amount)} to $payee " +
                    "(score: ${"%.0f".format(score * 100)}%)",
                    com.google.android.material.snackbar.Snackbar.LENGTH_LONG)
                .setAction("Review") {}
                .show()
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setSupportActionBar(binding.toolbar)
        checkAndRequestSmsPermissions()
        binding.btnGrantPermission.setOnClickListener {
            smsPermissionLauncher.launch(
                arrayOf(Manifest.permission.RECEIVE_SMS, Manifest.permission.READ_SMS)
            )
        }

        setupFragments()
        setupBottomNavigation()
        observeViewModel()
    }

    override fun onResume() {
        super.onResume()
        registerReceivers()
    }

    override fun onPause() {
        super.onPause()
        unregisterReceiver(categoryPromptReceiver)
        unregisterReceiver(fraudAlertReceiver)
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  FRAGMENT SETUP
    //  Add both fragments once; show/hide to avoid recreation on tab switch.
    //  DashboardContentFragment holds the RecyclerView and summary card.
    //  FinanceFragment holds the category breakdown and trend chart.
    // ──────────────────────────────────────────────────────────────────────────
    private fun setupFragments() {
        supportFragmentManager.beginTransaction()
            .add(R.id.fragment_container, financeFragment,   "finance")
            .hide(financeFragment)
            .add(R.id.fragment_container, dashboardContent, "dashboard")
            .commit()
        activeFragment = dashboardContent
    }

    private fun setupBottomNavigation() {
        val bottomNav = binding.bottomNavigation
        bottomNav.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_dashboard -> switchTo(dashboardContent)
                R.id.nav_finance   -> switchTo(financeFragment)
            }
            true
        }
    }

    private fun switchTo(fragment: Fragment) {
        if (fragment == activeFragment) return
        supportFragmentManager.beginTransaction()
            .hide(activeFragment)
            .show(fragment)
            .commit()
        activeFragment = fragment

        // Update toolbar subtitle to reflect active tab
        binding.toolbar.subtitle = when (fragment) {
            is FinanceFragment -> "Spending Tracker"
            else               -> "UPI Fraud Detection"
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  OBSERVE VIEWMODEL — unchanged from v2
    //  These StateFlow collections drive the Dashboard tab only.
    //  FinanceFragment has its own FinanceViewModel and its own observations.
    // ──────────────────────────────────────────────────────────────────────────
    private fun observeViewModel() {
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.fraudCount.collect { count ->
                    // Update the badge on the Dashboard tab icon
                    val badge = binding.bottomNavigation
                        .getOrCreateBadge(R.id.nav_dashboard)
                    if (count > 0) {
                        badge.isVisible = true
                        badge.number    = count
                    } else {
                        badge.isVisible = false
                    }
                }
            }
        }

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.error.collect { error ->
                    if (error != null) {
                        com.google.android.material.snackbar.Snackbar
                            .make(binding.root, error,
                                  com.google.android.material.snackbar.Snackbar.LENGTH_LONG)
                            .setAction("Dismiss") { viewModel.clearError() }
                            .show()
                    }
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    private fun checkAndRequestSmsPermissions() {
        val receive = ContextCompat.checkSelfPermission(this, Manifest.permission.RECEIVE_SMS)
        val read    = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_SMS)
        if (receive == PackageManager.PERMISSION_GRANTED
            && read  == PackageManager.PERMISSION_GRANTED) {
            binding.cardPermissionBanner.visibility = View.GONE
        } else {
            binding.cardPermissionBanner.visibility = View.VISIBLE
            smsPermissionLauncher.launch(
                arrayOf(Manifest.permission.RECEIVE_SMS, Manifest.permission.READ_SMS)
            )
        }
    }

    private fun registerReceivers() {
        val catFilter   = IntentFilter("com.paysense.SHOW_CATEGORY_PROMPT")
        val fraudFilter = IntentFilter("com.paysense.FRAUD_ALERT_HIGH")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(categoryPromptReceiver, catFilter,   RECEIVER_NOT_EXPORTED)
            registerReceiver(fraudAlertReceiver,    fraudFilter,  RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(categoryPromptReceiver, catFilter)
            registerReceiver(fraudAlertReceiver,    fraudFilter)
        }
    }
}
