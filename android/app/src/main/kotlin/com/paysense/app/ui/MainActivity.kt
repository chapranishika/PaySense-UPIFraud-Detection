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
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.paysense.app.R
import com.paysense.app.databinding.ActivityMainBinding
import com.paysense.app.layer2.PaySenseDatabase
import com.paysense.app.layer2.TransactionHistory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.NumberFormat
import java.util.Calendar
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var adapter: TransactionAdapter

    private val db by lazy { PaySenseDatabase.getInstance(this) }
    private val currencyFmt = NumberFormat.getCurrencyInstance(Locale("en", "IN"))

    private val smsPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        val allGranted = results.values.all { it }
        if (allGranted) {
            binding.cardPermissionBanner.visibility = View.GONE
        } else {
            binding.cardPermissionBanner.visibility = View.VISIBLE
        }
    }

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
            refreshDashboard()
            val payee = intent.getStringExtra("payee") ?: "Unknown"
            val amount = intent.getDoubleExtra("amount", 0.0)
            val score = intent.getDoubleExtra("fraud_score", 0.0)
            com.google.android.material.snackbar.Snackbar
                .make(binding.root, "🚨 High Risk: ₹$amount to $payee (${(score * 100).toInt()}%)", 2000)
                .show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupRecyclerView()
        checkAndRequestSmsPermissions()
        setupPermissionBannerButton()
        setupTestFraudButton()
        seedSampleData()
        refreshDashboard()
    }

    override fun onResume() {
        super.onResume()
        val filter = IntentFilter().apply {
            addAction("com.paysense.SHOW_CATEGORY_PROMPT")
            addAction("com.paysense.FRAUD_ALERT_HIGH")
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(categoryPromptReceiver, filter, RECEIVER_NOT_EXPORTED)
            registerReceiver(fraudAlertReceiver,    filter, RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(categoryPromptReceiver, filter)
            registerReceiver(fraudAlertReceiver,    filter)
        }
        refreshDashboard()
    }

    override fun onPause() {
        super.onPause()
        unregisterReceiver(categoryPromptReceiver)
        unregisterReceiver(fraudAlertReceiver)
    }

    private fun checkAndRequestSmsPermissions() {
        val permissions = mutableListOf(Manifest.permission.RECEIVE_SMS, Manifest.permission.READ_SMS)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        val allGranted = permissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
        binding.cardPermissionBanner.visibility = if (allGranted) View.GONE else View.VISIBLE
        if (!allGranted) smsPermissionLauncher.launch(permissions.toTypedArray())
    }

    private fun setupPermissionBannerButton() {
        binding.btnGrantPermission.setOnClickListener { checkAndRequestSmsPermissions() }
    }

    private fun setupTestFraudButton() {
        binding.btnTestFraud.setOnClickListener {
            lifecycleScope.launch {
                val mockTxn = withContext(Dispatchers.IO) {
                    val payees = listOf("Amazon", "Flipkart", "Swiggy", "Zomato", "Uber")
                    val amount = (100..50000).random().toDouble()
                    val isFraud = amount > 10000
                    val txn = TransactionHistory(
                        txnId = "DEMO_${System.currentTimeMillis()}",
                        payee = payees.random(),
                        amount = amount,
                        category = "Shopping",
                        senderId = "AD-PAYSN",
                        date = "Today",
                        timestamp = System.currentTimeMillis(),
                        fraudScore = if (isFraud) 0.85 else 0.1,
                        isFraud = isFraud,
                        alertLevel = if (isFraud) "high" else "none"
                    )
                    db.transactionDao().insertTransaction(txn)
                    txn
                }
                refreshDashboard()
                if (mockTxn.isFraud) {
                    com.google.android.material.snackbar.Snackbar.make(
                        binding.root, "🚨 High Risk Transaction Detected!", 3000
                    ).setBackgroundTint(ContextCompat.getColor(this@MainActivity, R.color.fraud_text)).show()
                }
            }
        }
    }

    private fun seedSampleData() {
        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                val dao = db.transactionDao()
                if (dao.getAllTransactions().isEmpty()) {
                    dao.insertTransaction(TransactionHistory("S1", "Zomato", 450.0, "Food", "AD-ZOMATO", "01 May", System.currentTimeMillis() - 86400000))
                    dao.insertTransaction(TransactionHistory("S2", "Amazon", 12500.0, "Shopping", "AD-AMAZON", "02 May", System.currentTimeMillis() - 43200000, 0.92, true, "high"))
                }
            }
            refreshDashboard()
        }
    }

    private fun setupRecyclerView() {
        adapter = TransactionAdapter()
        binding.rvTransactions.layoutManager = LinearLayoutManager(this)
        binding.rvTransactions.adapter = adapter
    }

    private fun refreshDashboard() {
        lifecycleScope.launch {
            val (list, spent) = withContext(Dispatchers.IO) {
                val dao = db.transactionDao()
                val monthStart = Calendar.getInstance().apply {
                    set(Calendar.DAY_OF_MONTH, 1)
                    set(Calendar.HOUR_OF_DAY, 0)
                    set(Calendar.MINUTE, 0)
                }.timeInMillis
                Pair(dao.getAllTransactions(), dao.getTotalSpentSince(monthStart))
            }
            adapter.submitList(list)
            binding.layoutEmptyState.visibility = if (list.isEmpty()) View.VISIBLE else View.GONE
            binding.tvTotalSpent.text = currencyFmt.format(spent)
            binding.tvTxnCount.text = list.size.toString()
            binding.tvFraudCount.text = list.count { it.isFraud }.toString()
        }
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(true)
    }
}
