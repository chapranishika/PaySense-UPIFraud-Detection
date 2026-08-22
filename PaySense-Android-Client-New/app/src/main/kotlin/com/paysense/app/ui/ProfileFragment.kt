package com.paysense.app.ui

import android.content.Context
import android.content.res.ColorStateList
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.google.android.material.snackbar.Snackbar
import com.paysense.app.R
import com.paysense.app.databinding.FragmentProfileBinding
import com.paysense.app.databinding.ItemProfileQuickLinkBinding

class ProfileFragment : Fragment() {

    private var _binding: FragmentProfileBinding? = null
    private val binding get() = _binding!!

    /**
     * Wireframe screen 5's quick-access rows. These sections don't have
     * dedicated destinations in this build yet, so tapping shows an honest
     * "coming soon" Snackbar rather than a dead tap or fabricated content.
     */
    private data class QuickLink(val label: String, val iconRes: Int, val chipColorRes: Int)

    private val quickLinks = listOf(
        QuickLink("Account Security",     R.drawable.ic_warning,       R.color.colorCoral),
        QuickLink("Payment Preferences",  R.drawable.ic_payment,       R.color.brand_primary),
        QuickLink("AI Insights",          R.drawable.ic_receipt_long,  R.color.colorGold),
        QuickLink("Notifications",        R.drawable.ic_sms,           R.color.colorPurple3),
        QuickLink("Help & Support",       R.drawable.ic_profile,       R.color.colorAccentDark),
        QuickLink("About PaySense",       R.drawable.ic_receipt,       R.color.colorMuted)
    )

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentProfileBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        setupQuickLinks()

        binding.btnProfileLogout.setOnClickListener {
            val prefs = requireContext().getSharedPreferences("paysense_prefs", Context.MODE_PRIVATE)
            prefs.edit().putBoolean("is_authenticated", false).apply()
            (activity as? MainActivity)?.showLoginOverlay()
        }
    }

    private fun setupQuickLinks() {
        val container = binding.layoutProfileQuickLinks
        container.removeAllViews()
        val inflater = LayoutInflater.from(requireContext())

        quickLinks.forEachIndexed { index, link ->
            val rowBinding = ItemProfileQuickLinkBinding.inflate(inflater, container, false)
            rowBinding.tvQuickLinkLabel.text = link.label
            rowBinding.ivQuickLinkIcon.setImageResource(link.iconRes)
            rowBinding.cardQuickLinkIcon.backgroundTintList =
                ColorStateList.valueOf(ContextCompat.getColor(requireContext(), link.chipColorRes))
            rowBinding.root.setOnClickListener {
                Snackbar.make(binding.root, "${link.label} — coming soon", Snackbar.LENGTH_SHORT).show()
            }
            container.addView(rowBinding.root)

            if (index != quickLinks.lastIndex) {
                val divider = View(requireContext()).apply {
                    layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, resources.displayMetrics.density.toInt())
                    setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.divider_color))
                }
                container.addView(divider)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    companion object {
        fun newInstance() = ProfileFragment()
    }
}
