package com.paysense.app.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
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

data class ChatMessage(val text: String, val isUser: Boolean)

class AssistantFragment : Fragment() {

    private var _binding: FragmentAssistantBinding? = null
    private val binding get() = _binding!!

    private val viewModel: PaySenseViewModel by activityViewModels()

    private val messages = mutableListOf<ChatMessage>()
    private lateinit var chatAdapter: ChatAdapter

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

    // Sends every message -- quick-action chips and real free-text alike --
    // to the backend's /assistant/chat, along with the user's real current
    // spend figures so the reply (real Gemini call or its guardrailed
    // fallback, decided entirely server-side) can ground any numbers it
    // states instead of inventing them. This replaced a client-side keyword
    // router that lived here before; see FraudApiService.chatWithAssistant()
    // and main.py's ASSISTANT_SYSTEM_INSTRUCTION for where the logic moved.
    private fun handleUserQuery(query: String) {
        addUserMessage(query)

        viewLifecycleOwner.lifecycleScope.launch {
            delay(800) // thinking delay, matches the assistant's prior feel

            val txns = viewModel.transactions.value
            val totalSpent = viewModel.totalSpent.value
            val fraudCount = viewModel.fraudCount.value

            val categories = txns
                .filter { !it.isFraud && it.category != "Income" && it.category != "Refund" }
                .groupBy { it.category }
            var topCategoryName = "Uncategorized"
            var topCategorySpent = 0.0
            for ((cat, catTxns) in categories) {
                val spent = catTxns.sumOf { it.amount }
                if (spent > topCategorySpent) {
                    topCategoryName = cat
                    topCategorySpent = spent
                }
            }
            val topCategoryPct = if (totalSpent > 0) (topCategorySpent / totalSpent) * 100 else 0.0

            val result = FraudApiService.getInstance(requireContext()).chatWithAssistant(
                message = query,
                totalSpent = totalSpent,
                topCategory = topCategoryName,
                topCategoryPct = topCategoryPct,
                fraudAlerts = fraudCount
            )

            if (_binding == null) return@launch // fragment view may be gone by the time this resolves

            addAssistantMessage(
                result?.reply
                    ?: "I can't reach PaySense right now — check your connection and try again."
            )
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
