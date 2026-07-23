package com.paysense.app.layer2

import android.content.Context
import android.content.Intent
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.pdf.PdfDocument
import androidx.core.content.FileProvider
import com.paysense.app.BuildConfig
import com.paysense.app.ui.BudgetProgress
import com.paysense.app.ui.BudgetStatus
import com.paysense.app.ui.MoMInsight
import com.paysense.app.ui.MoMTrend
import com.paysense.app.ui.SpendingPace
import java.io.File
import java.io.FileOutputStream
import java.text.NumberFormat
import java.util.Locale

/**
 * ============================================================================
 *  PdfReportGenerator.kt  — Phase 4: Monthly PDF Summary
 *
 *  Generates a single-page A4 PDF using android.graphics.pdf.PdfDocument
 *  (built into the Android SDK — no new Gradle dependency).
 *
 *  PAGE LAYOUT (A4 @ 72dpi = 595 x 842 pt):
 *    Header:   "PaySense Monthly Report — April 2025"
 *    Section 1: Summary card (total spent, avg/day, MoM change)
 *    Section 2: Category breakdown table with budget status
 *    Section 3: Top merchants
 *    Footer:   generated timestamp + disclaimer
 *
 *  WHY PdfDocument over a PDF library:
 *    PdfDocument is part of android.graphics — zero extra dependencies,
 *    zero APK size increase, and full control over layout via Canvas
 *    drawing primitives (drawText, drawRect, drawLine).
 *
 *  ZERO IMPACT ON FRAUD PIPELINE:
 *    Pure presentation layer — reads pre-computed ViewModel data, renders
 *    to PDF, returns a share Intent. No DB writes, no fraud code touched.
 * ============================================================================
 */
object PdfReportGenerator {

    private const val PAGE_WIDTH  = 595   // A4 width in points (72dpi)
    private const val PAGE_HEIGHT = 842   // A4 height in points
    private const val MARGIN      = 40f

    private val currFmt = NumberFormat.getCurrencyInstance(Locale("en", "IN"))

    // Colour palette matching the in-app dark theme
    private val NAVY  = Color.parseColor("#2D1B69")
    private val TEAL  = Color.parseColor("#00B4D8")
    private val GREEN = Color.parseColor("#43A047")
    private val AMBER = Color.parseColor("#F59E0B")
    private val RED   = Color.parseColor("#E53935")
    private val GREY  = Color.parseColor("#64748B")
    private val LIGHT = Color.parseColor("#EDE9FE")

    /**
     * Generates the PDF and returns a shareable Intent.
     *
     * @param monthLabel    e.g. "April 2025" — shown in the title
     * @param monthTotal    total spend this month
     * @param avgDaily      average daily spend
     * @param mom           month-over-month insight (nullable)
     * @param pace          spending pace projection (nullable)
     * @param budgetRows    category breakdown with budget status
     * @param topMerchants  top merchant list (payee, category, total, count)
     */
    fun generateMonthlyReport(
        context     : Context,
        monthLabel  : String,
        monthTotal  : Double,
        avgDaily    : Double,
        mom         : MoMInsight?,
        pace        : SpendingPace?,
        budgetRows  : List<BudgetProgress>,
        topMerchants: List<MerchantSpend>
    ): Intent? {
        return try {
            val doc  = PdfDocument()
            val page = doc.startPage(
                PdfDocument.PageInfo.Builder(PAGE_WIDTH, PAGE_HEIGHT, 1).create()
            )
            val canvas = page.canvas
            var y = MARGIN

            y = drawHeader(canvas, monthLabel, y)
            y = drawSummaryCard(canvas, monthTotal, avgDaily, mom, pace, y)
            y = drawCategoryTable(canvas, budgetRows, monthTotal, y)
            y = drawTopMerchants(canvas, topMerchants, y)
            drawFooter(canvas)

            doc.finishPage(page)

            val file = File(context.cacheDir, "PaySense_Report_${monthLabel.replace(" ", "_")}.pdf")
            FileOutputStream(file).use { doc.writeTo(it) }
            doc.close()

            val uri = FileProvider.getUriForFile(
                context, "${BuildConfig.APPLICATION_ID}.fileprovider", file
            )

            Intent(Intent.ACTION_SEND).apply {
                type = "application/pdf"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "PaySense Monthly Report — $monthLabel")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
        } catch (e: Exception) {
            null
        }
    }

    // ── HEADER ────────────────────────────────────────────────────────────────
    private fun drawHeader(canvas: Canvas, monthLabel: String, startY: Float): Float {
        var y = startY

        // Navy banner
        val bannerPaint = Paint().apply { color = NAVY }
        canvas.drawRect(0f, 0f, PAGE_WIDTH.toFloat(), 90f, bannerPaint)

        val titlePaint = Paint().apply {
            color = Color.WHITE; textSize = 22f; isFakeBoldText = true
            isAntiAlias = true
        }
        canvas.drawText("PaySense", MARGIN, 40f, titlePaint)

        val subPaint = Paint().apply {
            color = Color.parseColor("#A89CC8"); textSize = 12f; isAntiAlias = true
        }
        canvas.drawText("Monthly Spending Report — $monthLabel", MARGIN, 62f, subPaint)

        return 110f
    }

    // ── SUMMARY CARD ──────────────────────────────────────────────────────────
    private fun drawSummaryCard(
        canvas: Canvas, monthTotal: Double, avgDaily: Double,
        mom: MoMInsight?, pace: SpendingPace?, startY: Float
    ): Float {
        var y = startY
        val labelPaint = Paint().apply {
            color = GREY; textSize = 10f; isAntiAlias = true
        }
        val valuePaint = Paint().apply {
            color = NAVY; textSize = 18f; isFakeBoldText = true; isAntiAlias = true
        }

        canvas.drawText("TOTAL SPENT THIS MONTH", MARGIN, y, labelPaint)
        y += 22f
        canvas.drawText(currFmt.format(monthTotal), MARGIN, y, valuePaint)
        y += 28f

        // Three-column stats row
        val colWidth = (PAGE_WIDTH - 2 * MARGIN) / 3
        val statPaint = Paint().apply {
            color = TEAL; textSize = 14f; isFakeBoldText = true; isAntiAlias = true
        }

        canvas.drawText("Avg / day", MARGIN, y, labelPaint)
        canvas.drawText(currFmt.format(avgDaily), MARGIN, y + 18f, statPaint)

        mom?.let {
            val (momColor, momText) = when (it.trend) {
                MoMTrend.UP   -> RED   to "+%.1f%% vs last month".format(it.pctChange)
                MoMTrend.DOWN -> GREEN to "%.1f%% vs last month".format(it.pctChange)
                MoMTrend.FLAT -> GREY  to "Same as last month"
                MoMTrend.NO_DATA -> GREY to "No comparison data"
            }
            canvas.drawText("Month-over-month", MARGIN + colWidth, y, labelPaint)
            val momPaint = Paint(statPaint).apply { color = momColor }
            canvas.drawText(momText, MARGIN + colWidth, y + 18f, momPaint)
        }

        pace?.let {
            val (paceColor, paceText) = when (it.paceStatus) {
                com.paysense.app.ui.PaceStatus.ON_TRACK  -> GREEN to "On track"
                com.paysense.app.ui.PaceStatus.OVER_PACE -> RED   to "Over pace"
                com.paysense.app.ui.PaceStatus.NEUTRAL   -> AMBER to "On pace"
                com.paysense.app.ui.PaceStatus.NO_DATA   -> GREY  to "—"
            }
            canvas.drawText("Projected month-end", MARGIN + 2 * colWidth, y, labelPaint)
            val pacePaint = Paint(statPaint).apply { color = paceColor }
            canvas.drawText(currFmt.format(it.projectedTotal), MARGIN + 2 * colWidth, y + 18f, pacePaint)
            val tagPaint = Paint().apply { color = paceColor; textSize = 9f; isAntiAlias = true }
            canvas.drawText(paceText, MARGIN + 2 * colWidth, y + 32f, tagPaint)
        }

        y += 50f

        // Divider line
        val linePaint = Paint().apply { color = LIGHT; strokeWidth = 1f }
        canvas.drawLine(MARGIN, y, PAGE_WIDTH - MARGIN, y, linePaint)

        return y + 20f
    }

    // ── CATEGORY TABLE ────────────────────────────────────────────────────────
    private fun drawCategoryTable(
        canvas: Canvas, rows: List<BudgetProgress>, monthTotal: Double, startY: Float
    ): Float {
        var y = startY
        val headerPaint = Paint().apply {
            color = GREY; textSize = 10f; isFakeBoldText = true; isAntiAlias = true
        }
        canvas.drawText("CATEGORY BREAKDOWN", MARGIN, y, headerPaint)
        y += 20f

        val colCat    = MARGIN
        val colAmount = MARGIN + 180f
        val colPct    = MARGIN + 280f
        val colBudget = MARGIN + 360f

        // Table header row
        val thPaint = Paint().apply { color = GREY; textSize = 9f; isAntiAlias = true }
        canvas.drawText("Category",  colCat,    y, thPaint)
        canvas.drawText("Amount",    colAmount, y, thPaint)
        canvas.drawText("% of total",colPct,    y, thPaint)
        canvas.drawText("Budget status", colBudget, y, thPaint)
        y += 6f
        val linePaint = Paint().apply { color = LIGHT; strokeWidth = 1f }
        canvas.drawLine(MARGIN, y, PAGE_WIDTH - MARGIN, y, linePaint)
        y += 18f

        val namePaint = Paint().apply { color = NAVY; textSize = 11f; isAntiAlias = true }
        val amtPaint  = Paint().apply { color = NAVY; textSize = 11f; isFakeBoldText = true; isAntiAlias = true }
        val pctPaint  = Paint().apply { color = GREY; textSize = 10f; isAntiAlias = true }

        rows.take(12).forEach { bp ->
            val cat = bp.categorySpend
            val pct = if (monthTotal > 0) (cat.total / monthTotal * 100) else 0.0

            canvas.drawText(cat.category, colCat, y, namePaint)
            canvas.drawText(currFmt.format(cat.total), colAmount, y, amtPaint)
            canvas.drawText("%.1f%%".format(pct), colPct, y, pctPaint)

            // Budget status badge
            val (statusColor, statusText) = when (bp.status) {
                BudgetStatus.OVER    -> RED   to "Over budget"
                BudgetStatus.WARNING -> AMBER to "%.0f%% used".format(bp.pctUsed ?: 0f)
                BudgetStatus.OK      -> GREEN to "%.0f%% used".format(bp.pctUsed ?: 0f)
                BudgetStatus.NONE    -> GREY  to "No budget set"
            }
            val statusPaint = Paint().apply {
                color = statusColor; textSize = 10f; isAntiAlias = true
            }
            canvas.drawText(statusText, colBudget, y, statusPaint)

            y += 20f
        }

        return y + 10f
    }

    // ── TOP MERCHANTS ─────────────────────────────────────────────────────────
    private fun drawTopMerchants(canvas: Canvas, merchants: List<MerchantSpend>, startY: Float): Float {
        var y = startY
        val headerPaint = Paint().apply {
            color = GREY; textSize = 10f; isFakeBoldText = true; isAntiAlias = true
        }
        canvas.drawText("TOP MERCHANTS", MARGIN, y, headerPaint)
        y += 20f

        val namePaint = Paint().apply { color = NAVY; textSize = 11f; isAntiAlias = true }
        val amtPaint  = Paint().apply { color = NAVY; textSize = 11f; isFakeBoldText = true; isAntiAlias = true }
        val subPaint  = Paint().apply { color = GREY; textSize = 9f; isAntiAlias = true }

        merchants.take(5).forEachIndexed { idx, m ->
            canvas.drawText("${idx + 1}.", MARGIN, y, namePaint)
            canvas.drawText(m.payee, MARGIN + 20f, y, namePaint)
            canvas.drawText(currFmt.format(m.total), PAGE_WIDTH - MARGIN - 80f, y, amtPaint)
            y += 14f
            canvas.drawText("${m.txnCount} visits · ${m.category}", MARGIN + 20f, y, subPaint)
            y += 18f
        }
        return y + 10f
    }

    // ── FOOTER ────────────────────────────────────────────────────────────────
    private fun drawFooter(canvas: Canvas) {
        val footerPaint = Paint().apply {
            color = GREY; textSize = 8f; isAntiAlias = true
        }
        val sdf = java.text.SimpleDateFormat("dd MMM yyyy, hh:mm a", Locale("en", "IN"))
        val timestamp = sdf.format(java.util.Date())
        canvas.drawText(
            "Generated by PaySense on $timestamp · Fraud-flagged transactions excluded from totals",
            MARGIN, PAGE_HEIGHT - 20f, footerPaint
        )
    }
}
