package com.paysense.app.ui

import android.content.Context
import android.content.res.ColorStateList
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import com.paysense.app.R
import com.paysense.app.databinding.FragmentProfileBinding
import com.paysense.app.databinding.ItemProfileQuickLinkBinding
import com.paysense.app.layer3.FraudApiService
import com.paysense.app.layer3.SecurePrefs
import kotlinx.coroutines.launch

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
        loadLiveEnsembleConfig()

        binding.btnProfileLogout.setOnClickListener {
            val prefs = SecurePrefs.get(requireContext())
            // Also remove auth_token, not just flip is_authenticated -- otherwise
            // a still-valid JWT (up to ~60 min server-side) sits in prefs after
            // the user believes they've logged out. Matches FraudApiService's
            // own clearAuth(), used on the 401 path.
            prefs.edit()
                .putBoolean("is_authenticated", false)
                .remove("auth_token")
                .apply()
            (activity as? MainActivity)?.showLoginOverlay()
        }
    }

    // Populates Decision Threshold / Ensemble Weights from a real GET /health
    // call instead of the hardcoded "0.4000 / Rules(0.15)*XGB(0.85)" text
    // this screen originally shipped with -- stale since before this
    // session's threshold corrections and before the real 3-scorer ensemble
    // existed. Same fix, same reasoning, as the web dashboard's Profile page
    // (see main.py's /health `nominal_weights` field). Leaves the existing
    // placeholder text in place on any failure rather than showing an error.
    private fun loadLiveEnsembleConfig() {
        viewLifecycleOwner.lifecycleScope.launch {
            val health = FraudApiService.getInstance(requireContext()).getHealth() ?: return@launch
            val threshold = (health["threshold"] as? Number)?.toDouble()
            @Suppress("UNCHECKED_CAST")
            val weights = health["nominal_weights"] as? Map<String, Any>

            if (_binding == null) return@launch  // fragment view may be gone by the time this resolves
            if (threshold != null) {
                binding.tvDecisionThreshold.text = String.format("%.4f", threshold)
            }
            if (weights != null) {
                val rules = (weights["rules"] as? Number)?.toDouble()
                val paysense = (weights["paysense"] as? Number)?.toDouble()
                val lightLr = (weights["light_lr"] as? Number)?.toDouble()
                if (rules != null && paysense != null && lightLr != null) {
                    binding.tvEnsembleWeights.text =
                        "Rules ($rules) · XGBoost ($paysense) · LightLR ($lightLr)"
                }
            }
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
