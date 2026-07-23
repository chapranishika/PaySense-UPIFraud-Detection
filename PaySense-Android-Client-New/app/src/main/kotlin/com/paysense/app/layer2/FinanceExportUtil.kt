package com.paysense.app.layer2

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import com.paysense.app.BuildConfig
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * ============================================================================
 *  FinanceExportUtil.kt  — Phase 4: CSV Export
 *
 *  Exports transaction_history rows to a CSV file in the app's cache
 *  directory, then returns a content:// URI via FileProvider that can be
 *  shared through any Intent.ACTION_SEND target (email, Drive, WhatsApp).
 *
 *  WHY FileProvider:
 *    Since Android 10, apps cannot share file:// URIs with other apps —
 *    it throws FileUriExposedException. FileProvider wraps the file in a
 *    content:// URI with a temporary read permission grant to the
 *    receiving app. Requires a <provider> entry in AndroidManifest.xml
 *    and a res/xml/file_paths.xml declaring which directories are shareable.
 *
 *  CSV SCHEMA (one row per transaction):
 *    Date, Payee, Category, Amount, Fraud Score, Alert Level
 *
 *  ZERO IMPACT ON FRAUD PIPELINE:
 *    Read-only — exports a copy of transaction_history. Does not modify
 *    any table, does not touch SmsReceiver/FraudApiService.
 * ============================================================================
 */
object FinanceExportUtil {

    private val dateFmt = SimpleDateFormat("dd MMM yyyy, hh:mm a", Locale("en", "IN"))

    /**
     * Writes all given transactions to a CSV file and returns a shareable
     * Intent. Call startActivity() on the result.
     *
     * @param context     Application or Activity context
     * @param transactions list of TransactionHistory rows to export (already
     *                     filtered by the caller — e.g. current month only)
     * @param fileLabel   filename suffix, e.g. "April_2025"
     * @return Intent.ACTION_SEND chooser intent, or null if write failed
     */
    fun exportToCsv(
        context: Context,
        transactions: List<TransactionHistory>,
        fileLabel: String
    ): Intent? {
        return try {
            val file = File(context.cacheDir, "PaySense_Transactions_$fileLabel.csv")
            FileWriter(file).use { writer ->
                // Header row
                writer.append("Date,Payee,Category,Amount (INR),Fraud Score,Alert Level\n")

                transactions.forEach { txn ->
                    writer.append(escapeCsv(dateFmt.format(Date(txn.timestamp))))
                    writer.append(",")
                    writer.append(escapeCsv(txn.payee))
                    writer.append(",")
                    writer.append(escapeCsv(txn.category))
                    writer.append(",")
                    writer.append(String.format(Locale.US, "%.2f", txn.amount))
                    writer.append(",")
                    writer.append(String.format(Locale.US, "%.4f", txn.fraudScore))
                    writer.append(",")
                    writer.append(escapeCsv(txn.alertLevel))
                    writer.append("\n")
                }
            }

            val uri = FileProvider.getUriForFile(
                context,
                "${BuildConfig.APPLICATION_ID}.fileprovider",
                file
            )

            Intent(Intent.ACTION_SEND).apply {
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "PaySense Transactions — $fileLabel")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        } catch (e: Exception) {
            null
        }
    }

    /**
     * CSV field escaping per RFC 4180:
     *   - Wrap in quotes if the field contains a comma, quote, or newline
     *   - Double any embedded quote characters
     */
    private fun escapeCsv(field: String): String {
        return if (field.contains(",") || field.contains("\"") || field.contains("\n")) {
            "\"${field.replace("\"", "\"\"")}\""
        } else field
    }
}
