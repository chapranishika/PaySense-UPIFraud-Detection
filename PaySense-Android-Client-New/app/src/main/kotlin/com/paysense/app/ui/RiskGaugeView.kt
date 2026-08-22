package com.paysense.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.paysense.app.R

/**
 * ============================================================================
 *  RiskGaugeView — a single-arc circular ring gauge.
 *
 *  Wireframe pass: the reference wireframe uses the same "ring with a
 *  partially-filled coloured arc + centred number" pattern on two screens —
 *  the dashboard's "Security Status" safety ring and the "Risk Details"
 *  score gauge. Rather than build two bespoke views, this one reusable View
 *  draws just the ring (track + progress arc); the centre number/label is a
 *  plain overlaid TextView in the layout, which keeps text rendering
 *  Poppins-consistent with the rest of the screen instead of drawing text
 *  on a Canvas.
 * ============================================================================
 */
class RiskGaugeView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        color = ContextCompat.getColor(context, R.color.card_stroke_normal)
    }

    private val progressPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        color = ContextCompat.getColor(context, R.color.colorAlertNone)
    }

    private val rect = RectF()

    /** Fraction of the ring to fill, 0f..1f. Clamped defensively. */
    var progress: Float = 0f
        set(value) {
            field = value.coerceIn(0f, 1f)
            invalidate()
        }

    fun setProgressColor(color: Int) {
        progressPaint.color = color
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        val size = minOf(w, h)
        if (size <= 0f) return

        val stroke = size * 0.11f
        val padding = stroke / 2f + 4f

        rect.set(
            (w - size) / 2f + padding,
            (h - size) / 2f + padding,
            (w + size) / 2f - padding,
            (h + size) / 2f - padding
        )

        trackPaint.strokeWidth = stroke
        progressPaint.strokeWidth = stroke

        // Full track ring first, then the coloured progress arc on top,
        // starting at 12 o'clock (-90deg) going clockwise.
        canvas.drawArc(rect, 0f, 360f, false, trackPaint)
        if (progress > 0f) {
            canvas.drawArc(rect, -90f, progress * 360f, false, progressPaint)
        }
    }
}
