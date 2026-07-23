package com.paysense.app.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.paysense.app.databinding.FragmentAssistantBinding
import com.paysense.app.databinding.ItemChatBubbleBinding
import com.paysense.app.layer3.FraudApiService
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.text.NumberFormat
import java.util.Locale

data class ChatMessage(val text: String, val isUser: Boolean)

class AssistantFragment : Fragment() {

    private var _binding: FragmentAssistantBinding? = null
    private val binding get() = _binding!!

    private val viewModel: PaySenseViewModel by activityViewModels()

    private val messages = mutableListOf<ChatMessage>()
    private lateinit var chatAdapter: ChatAdapter
    private val currencyFmt = NumberFormat.getCurrencyInstance(Locale("en", "IN"))

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentAssistantBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupRecyclerView()
        setupChatbot()

        // Welcome message if chat is empty
        if (messages.isEmpty()) {
            addAssistantMessage("Hi Nishika! I am your PaySense Finlatics Assistant. I monitor your local transaction history and XGBoost model parameters to help you save money and stay secure. Ask me for a \"summary\" or a \"savings tip\"!")
        }
    }

    private fun setupRecyclerView() {
        chatAdapter = ChatAdapter(messages)
        binding.rvChatMessages.apply {
            layoutManager = LinearLayoutManager(requireContext()).apply {
                stackFromEnd = true // scroll to bottom as new messages arrive
            }
            adapter = chatAdapter
        }
    }

    private fun setupChatbot() {
        binding.btnChatSend.setOnClickListener {
            val query = binding.etChatQuery.text.toString().trim()
            if (query.isNotEmpty()) {
                handleUserQuery(query)
                binding.etChatQuery.text?.clear()
            }
        }

        binding.btnQuickSummary.setOnClickListener {
            handleUserQuery("Give me a spending summary")
        }

        binding.btnQuickTip.setOnClickListener {
            handleUserQuery("Give me a savings tip")
        }

        binding.btnQuickSecurity.setOnClickListener {
            handleUserQuery("Check my fraud status")
        }
    }

    private fun handleUserQuery(query: String) {
        // 1. Add user message
        addUserMessage(query)

        // 2. Launch coroutine to generate reply with simulated thinking delay
        viewLifecycleOwner.lifecycleScope.launch {
            delay(800) // 800ms thinking delay

            val lower = query.lowercase(Locale.ROOT)
            when {
                lower.contains("summary") || lower.contains("spend") -> {
                    val summaryText = generateSpendingSummary()
                    addAssistantMessage(summaryText)
                }
                lower.contains("tip") || lower.contains("save") || lower.contains("dining") -> {
                    // Call live insights API!
                    val tipText = fetchSavingsTipFromApi()
                    addAssistantMessage(tipText)
                }
                lower.contains("fraud") || lower.contains("security") || lower.contains("status") || lower.contains("alert") -> {
                    val fraudStatusText = generateFraudStatusReport()
                    addAssistantMessage(fraudStatusText)
                }
                else -> {
                    addAssistantMessage("I can analyze your spending history. Try typing:\n- \"spending summary\"\n- \"savings tip\"\n- \"fraud status\"")
                }
            }
        }
    }

    private fun addUserMessage(text: String) {
        messages.add(ChatMessage(text, isUser = true))
        chatAdapter.notifyItemInserted(messages.size - 1)
        binding.rvChatMessages.scrollToPosition(messages.size - 1)
    }

    private fun addAssistantMessage(text: String) {
        messages.add(ChatMessage(text, isUser = false))
        chatAdapter.notifyItemInserted(messages.size - 1)
        binding.rvChatMessages.scrollToPosition(messages.size - 1)
    }

    private fun generateSpendingSummary(): String {
        val txns = viewModel.transactions.value
        val totalSpent = viewModel.totalSpent.value
        val fraudCount = viewModel.fraudCount.value

        if (txns.isEmpty()) {
            return "You don't have any transaction history yet. Type or simulate an SMS to get started!"
        }

        // Aggregate category counts
        val categories = txns.filter { !it.isFraud }.groupBy { it.category }
        var topCategoryName = "Uncategorized"
        var topCategorySpent = 0.0
        
        for ((cat, catTxns) in categories) {
            val spent = catTxns.sumOf { it.amount }
            if (spent > topCategorySpent) {
                topCategoryName = cat
                topCategorySpent = spent
            }
        }

        return "📊 **Spending Summary:**\n\n" +
                "- **Total Outflow:** ${currencyFmt.format(totalSpent)}\n" +
                "- **Top Category:** $topCategoryName (${currencyFmt.format(topCategorySpent)})\n" +
                "- **Fraud Attempts Blocked:** $fraudCount hits\n\n" +
                "Your weekly budget utilization is at **${((totalSpent / 15000.0) * 100).toInt()}%** of your ₹15,000 threshold."
    }

    private suspend fun fetchSavingsTipFromApi(): String {
        val totalSpent = viewModel.totalSpent.value
        val fraudCount = viewModel.fraudCount.value
        val txns = viewModel.transactions.value

        // Find top category
        val categories = txns.filter { !it.isFraud }.groupBy { it.category }
        var topCategoryName = "Food"
        var topCategorySpent = 0.0
        for ((cat, catTxns) in categories) {
            val spent = catTxns.sumOf { it.amount }
            if (spent > topCategorySpent) {
                topCategoryName = cat
                topCategorySpent = spent
            }
        }
        val topCategoryPct = if (totalSpent > 0) (topCategorySpent / totalSpent) * 100 else 0.0

        // Call backend API
        val service = FraudApiService.getInstance(requireContext())
        val insight = service.getWeeklyInsights(
            totalSpent = totalSpent,
            topCategory = topCategoryName,
            topCategoryPct = topCategoryPct,
            fraudAlerts = fraudCount
        )

        return if (insight != null) {
            "💡 **AI Savings Tip:**\n\n\"${insight.savingsTip}\"\n\n*Budget Pace: ${insight.budgetStatus}*"
        } else {
            // Local fallback tip
            "💡 **Savings Tip:**\n\nCooking meals at home instead of ordering out can save up to ₹800/week on food expenses."
        }
    }

    private fun generateFraudStatusReport(): String {
        val fraudCount = viewModel.fraudCount.value
        return "🛡️ **PaySense Security Status:**\n\n" +
                "- **Active Engine:** XGBoost 41-feature Ensemble Scorer.\n" +
                "- **Layer 1 Gate:** TRAI format check active.\n" +
                "- **Blocked Incidents:** $fraudCount fraud attempts intercepted.\n" +
                "- **Current Threat Level:** LEGIT. Any transaction scoring above 0.70 will trigger immediate block alerts."
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    companion object {
        fun newInstance() = AssistantFragment()
    }

    // ── Chat Bubble Recycler Adapter ──────────────────────────────────────────
    private class ChatAdapter(private val messages: List<ChatMessage>) :
        RecyclerView.Adapter<ChatAdapter.ChatViewHolder>() {

        class ChatViewHolder(val binding: ItemChatBubbleBinding) : RecyclerView.ViewHolder(binding.root)

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ChatViewHolder {
            val binding = ItemChatBubbleBinding.inflate(
                LayoutInflater.from(parent.context), parent, false
            )
            return ChatViewHolder(binding)
        }

        override fun onBindViewHolder(holder: ChatViewHolder, position: Int) {
            val msg = messages[position]
            if (msg.isUser) {
                holder.binding.layoutUser.visibility = View.VISIBLE
                holder.binding.layoutAssistant.visibility = View.GONE
                holder.binding.tvUserMessage.text = msg.text
            } else {
                holder.binding.layoutUser.visibility = View.GONE
                holder.binding.layoutAssistant.visibility = View.VISIBLE
                holder.binding.tvAssistantMessage.text = msg.text
            }
        }

        override fun getItemCount(): Int = messages.size
    }
}
