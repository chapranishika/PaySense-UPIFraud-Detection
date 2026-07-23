package com.paysense.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class DonutChartView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }

    private val rect = RectF()
    private var data: List<Pair<Float, Int>> = emptyList()

    fun setData(newData: List<Pair<Float, Int>>) {
        this.data = newData
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (data.isEmpty()) return

        val w = width.toFloat()
        val h = height.toFloat()
        val size = Math.min(w, h)
        val padding = size * 0.15f
        val stroke = size * 0.12f

        rect.set(
            (w - size) / 2f + padding,
            (h - size) / 2f + padding,
            (w + size) / 2f - padding,
            (h + size) / 2f - padding
        )
        
        paint.strokeWidth = stroke

        var startAngle = -90f
        val total = data.sumOf { it.first.toDouble() }.toFloat()
        if (total <= 0f) return

        for (item in data) {
            val sweepAngle = (item.first / total) * 360f
            paint.color = item.second
            canvas.drawArc(rect, startAngle + 2f, sweepAngle - 4f, false, paint)
            startAngle += sweepAngle
        }
    }
}
