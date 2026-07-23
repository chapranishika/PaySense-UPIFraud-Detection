package com.paysense.app.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.fragment.app.DialogFragment
import androidx.fragment.app.activityViewModels
import com.paysense.app.databinding.DialogAddTransactionBinding

class AddTransactionDialog : DialogFragment() {

    private var _binding: DialogAddTransactionBinding? = null
    private val binding get() = _binding!!

    private val viewModel: PaySenseViewModel by activityViewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Make the dialog fit screen width nicely
        setStyle(STYLE_NO_TITLE, android.R.style.Theme_DeviceDefault_Dialog_MinWidth)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = DialogAddTransactionBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        setupSpinners()
        setupButtons()
    }

    private fun setupSpinners() {
        val categories = arrayOf("Food & Dining", "Shopping", "Grocery", "Travel", "Entertainment", "Uncategorized")
        val apps = arrayOf("GPay", "PhonePe", "Paytm")
        val devices = arrayOf("Android", "iOS")

        binding.spCategory.adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_dropdown_item, categories)
        binding.spApp.adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_dropdown_item, apps)
        binding.spDevice.adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_dropdown_item, devices)
    }

    private fun setupButtons() {
        binding.btnCancel.setOnClickListener {
            dismiss()
        }

        binding.btnSubmit.setOnClickListener {
            val payee = binding.etPayee.text.toString().trim()
            val amountStr = binding.etAmount.text.toString().trim()
            val hourStr = binding.etHour.text.toString().trim()

            if (payee.isEmpty() || amountStr.isEmpty()) {
                Toast.makeText(requireContext(), "Please fill in payee and amount", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val amount = amountStr.toDoubleOrNull()
            if (amount == null || amount <= 0) {
                Toast.makeText(requireContext(), "Please enter a valid amount", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val hour = hourStr.toIntOrNull() ?: 12
            if (hour !in 0..23) {
                Toast.makeText(requireContext(), "Hour must be between 0 and 23", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val category = binding.spCategory.selectedItem.toString()
            val app = binding.spApp.selectedItem.toString()
            val device = binding.spDevice.selectedItem.toString()
            val newDevice = if (binding.cbNewDevice.isChecked) 1 else 0
            val ipMismatch = if (binding.cbIpMismatch.isChecked) 1 else 0

            // Trigger ViewModel logic
            viewModel.addManualTransaction(
                payee = payee,
                amount = amount,
                category = category,
                app = app,
                device = device,
                hour = hour,
                newDevice = newDevice,
                ipMismatch = ipMismatch
            )

            Toast.makeText(requireContext(), "Scoring transaction...", Toast.LENGTH_SHORT).show()
            dismiss()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    companion object {
        const val TAG = "AddTransactionDialog"
        fun newInstance() = AddTransactionDialog()
    }
}
