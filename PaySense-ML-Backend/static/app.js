/* ==========================================================================
   PaySense 2.0 Client-Side Orchestrator & AI Engine
   ========================================================================== */

const CONFIG = {
    apiBase: "", // Relative to current host
    defaultUser: {
        username: "paysense",
        name: "Nishika Chapra",
        ageGroup: "25-34",
        cityTier: 2,
        loyaltyScore: 0.85,
        avgMonthlyTxn: 15,
        avgTxnValue: 500,
        linkedBanks: 1.0,
        accountAgeDays: 365,
        preferredApp: "GPay",
        preferredDevice: "Android"
    }
};

// Application State
let state = {
    token: localStorage.getItem("paysense_token") || null,
    transactions: [
        {
            payee: "Zomato India",
            amount: 500.00,
            category: "Food & Dining",
            app: "GPay",
            time: "10:14 AM",
            isFraud: false,
            fraudScore: 0.08,
            device: "Android"
        },
        {
            payee: "Amazon India",
            amount: 1200.00,
            category: "Shopping",
            app: "GPay",
            time: "Yesterday",
            isFraud: false,
            fraudScore: 0.12,
            device: "Android"
        }
    ],
    fraudAlerts: 0,
    monthlySpent: 1700.00,
    activeTab: "overview",
    healthInterval: null
};

// Initialize app when DOM loads
document.addEventListener("DOMContentLoaded", () => {
    initAuth();
    initNavigation();
    initModal();
    initTestingControls();
    initChatbot();
});

// ==========================================================================
//  Authentication Flow
// ==========================================================================
function initAuth() {
    const loginForm = document.getElementById("login-form");
    const loginError = document.getElementById("login-error");
    const logoutBtn = document.getElementById("logout-btn");

    if (state.token) {
        showScreen("dashboard-screen");
        startDashboard();
    } else {
        showScreen("auth-screen");
    }

    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        loginError.classList.add("hide");
        const usernameInput = document.getElementById("username").value.trim();
        const passwordInput = document.getElementById("password").value.trim();

        try {
            const response = await fetch(`${CONFIG.apiBase}/auth/token`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username: usernameInput, password: passwordInput })
            });

            const data = await response.json();
            
            if (response.ok) {
                state.token = data.access_token;
                localStorage.setItem("paysense_token", data.access_token);
                showScreen("dashboard-screen");
                startDashboard();
            } else {
                loginError.textContent = data.detail || "Authentication failed. Try again.";
                loginError.classList.remove("hide");
            }
        } catch (err) {
            loginError.textContent = "Server offline. Ensure FastAPI is running.";
            loginError.classList.remove("hide");
        }
    });

    logoutBtn.addEventListener("click", (e) => {
        e.preventDefault();
        state.token = null;
        localStorage.removeItem("paysense_token");
        clearInterval(state.healthInterval);
        showScreen("auth-screen");
    });
}

function showScreen(screenId) {
    document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
    document.getElementById(screenId).classList.add("active");
}

function startDashboard() {
    checkHealth();
    state.healthInterval = setInterval(checkHealth, 10000);
    renderTransactions();
    renderFinance();
}

// ==========================================================================
//  Tab Navigation
// ==========================================================================
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item[data-tab]");
    const tabContents = document.querySelectorAll(".tab-content");
    const activeTitle = document.getElementById("active-tab-title");
    const activeSubtitle = document.getElementById("active-tab-subtitle");

    const tabMetadata = {
        "tab-overview": { title: "Overview", subtitle: "Real-time fraud defense baselines and transaction intercept logs" },
        "tab-finance": { title: "Finance Tracker", subtitle: "Personal budget optimization and monthly category analytics" },
        "tab-assistant": { title: "AI Guardian Assistant", subtitle: "Deep coaching insights and personalized savings advisor" },
        "tab-profile": { title: "User Profile", subtitle: "Current scoring profile parameters and model threshold configurations" }
    };

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const tabId = item.getAttribute("data-tab");
            
            navItems.forEach(n => n.classList.remove("active"));
            item.classList.add("active");

            tabContents.forEach(content => {
                if (content.id === tabId) {
                    content.classList.add("active");
                } else {
                    content.classList.remove("active");
                }
            });

            const meta = tabMetadata[tabId] || { title: "Overview", subtitle: "" };
            activeTitle.textContent = meta.title;
            activeSubtitle.textContent = meta.subtitle;
            
            if (tabId === "tab-finance") renderFinance();
        });
    });
}

// ==========================================================================
//  System Health Checks
// ==========================================================================
async function checkHealth() {
    const healthStatus = document.getElementById("health-status");
    const healthModel = document.getElementById("health-model");
    
    try {
        const response = await fetch(`${CONFIG.apiBase}/health`);
        const data = await response.json();
        
        if (response.ok && data.status === "ok") {
            healthStatus.textContent = "Online";
            healthStatus.className = "badge badge-online";
            healthModel.textContent = `XGBoost (${data.feature_count} features)`;
        } else {
            healthStatus.textContent = "Error";
            healthStatus.className = "badge badge-offline";
            healthModel.textContent = "None";
        }
    } catch (err) {
        healthStatus.textContent = "Offline";
        healthStatus.className = "badge badge-offline";
        healthModel.textContent = "None";
    }
}

// ==========================================================================
//  Modal Manual Input & Scoring
// ==========================================================================
function initModal() {
    const modal = document.getElementById("add-spent-modal");
    const btnOpen = document.getElementById("btn-open-modal");
    const btnClose = document.getElementById("btn-close-modal");
    const btnCancel = document.getElementById("btn-cancel-modal");
    const form = document.getElementById("add-transaction-form");

    btnOpen.addEventListener("click", () => modal.classList.add("active"));
    
    const closeModal = () => {
        modal.classList.remove("active");
        form.reset();
    };

    btnClose.addEventListener("click", closeModal);
    btnCancel.addEventListener("click", closeModal);

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const payee = document.getElementById("input-payee").value.trim();
        const amount = parseFloat(document.getElementById("input-amount").value);
        const category = document.getElementById("input-category").value;
        const app = document.getElementById("input-app").value;
        const device = document.getElementById("input-device").value;
        const hour = parseInt(document.getElementById("input-hour").value);
        const newDevice = document.getElementById("input-new-device").checked ? 1 : 0;
        const ipMismatch = document.getElementById("input-ip-mismatch").checked ? 1 : 0;

        await scoreAndSaveTransaction({ payee, amount, category, app, device, hour, newDevice, ipMismatch });
        closeModal();
    });
}

// Main logic to call /predict on FastAPI backend
async function scoreAndSaveTransaction({ payee, amount, category, app, device, hour, newDevice, ipMismatch }) {
    // 1. Calculate amount z-score deviation relative to default baseline profile
    const zScore = (amount - CONFIG.defaultUser.avgTxnValue) / 150.0; // standard deviation fallback = 150
    
    // 2. Build full 41-feature XGBoost payload matching the model expectations
    const payload = {
        receiver_type: payee.toLowerCase().includes("amazon") || payee.toLowerCase().includes("zomato") ? "Merchant" : "Individual",
        transaction_type: payee.toLowerCase().includes("amazon") || payee.toLowerCase().includes("zomato") ? "P2M" : "P2P",
        payment_app: app,
        device_type: device,
        usr_age_group: CONFIG.defaultUser.ageGroup,
        usr_preferred_app: CONFIG.defaultUser.preferredApp,
        usr_preferred_device: CONFIG.defaultUser.preferredDevice,
        mrc_category: category,
        mrc_size: amount > 5000 ? "Large" : "Medium",
        amount: amount,
        hour_of_day: hour,
        day_of_week: new Date().getDay(),
        is_weekend: new Date().getDay() === 0 || new Date().getDay() === 6 ? 1 : 0,
        is_night_transaction: hour >= 22 || hour <= 5 ? 1 : 0,
        time_since_last_txn_min: 15.0,
        transaction_velocity: 0.5,
        amount_deviation_score: Math.max(0.0, zScore),
        failed_attempts_last_24h: 0.0,
        recurring_payment_flag: 0,
        transaction_frequency_score: 0.8,
        new_device_flag: newDevice,
        ip_location_mismatch: ipMismatch,
        user_city_tier: CONFIG.defaultUser.cityTier,
        user_avg_monthly_txn: parseFloat(CONFIG.defaultUser.avgMonthlyTxn),
        user_avg_txn_value: parseFloat(CONFIG.defaultUser.avgTxnValue),
        user_loyalty_score: parseFloat(CONFIG.defaultUser.loyaltyScore),
        balance_after_transaction: 12000.0 - amount,
        txn_success_flag: 1,
        kyc_verified_flag: 1,
        usr_home_city_tier: CONFIG.defaultUser.cityTier,
        usr_account_age_days: parseFloat(CONFIG.defaultUser.accountAgeDays),
        usr_linked_bank_count: parseFloat(CONFIG.defaultUser.linkedBanks),
        usr_avg_monthly_txn_profile: parseFloat(CONFIG.defaultUser.avgMonthlyTxn),
        usr_avg_txn_value_profile: parseFloat(CONFIG.defaultUser.avgTxnValue),
        usr_is_high_risk: 0,
        mrc_avg_daily_txn: 250.0,
        mrc_is_registered: 1,
        mrc_rating: 4.2,
        device_risk_score: newDevice ? 0.9 : 0.1,
        ip_risk_score: ipMismatch ? 0.95 : 0.15
    };

    let isFraud = false;
    let fraudScore = 0.05;

    // Call live API
    try {
        const response = await fetch(`${CONFIG.apiBase}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${state.token}`
            },
            body: JSON.stringify(payload)
        });
        
        if (response.ok) {
            const data = await response.json();
            isFraud = data.is_fraud;
            fraudScore = data.fraud_score;
        }
    } catch (err) {
        console.warn("Prediction fallback active (using mock rules due to backend offline)");
        // Fallback mock logic if server is offline
        if (amount > 10000 || newDevice || ipMismatch) {
            isFraud = true;
            fraudScore = 0.88;
        }
    }

    // Add transaction to state
    const newTxn = {
        payee,
        amount,
        category,
        app,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isFraud,
        fraudScore,
        device
    };

    state.transactions.unshift(newTxn);
    
    if (isFraud) {
        state.fraudAlerts++;
        triggerAlertNotification(payee, amount, fraudScore);
    } else {
        state.monthlySpent += amount;
    }

    renderTransactions();
    renderFinance();
}

function triggerAlertNotification(payee, amount, score) {
    const alertIconWrapper = document.getElementById("alert-icon-wrapper");
    // Add brief flash animation effect to alert card
    alertIconWrapper.style.animation = "flashAlert 0.5s ease 3";
    setTimeout(() => { alertIconWrapper.style.animation = ""; }, 1500);
}

// Render dynamic transaction logs
function renderTransactions() {
    const feed = document.getElementById("transaction-feed");
    const fraudCounter = document.getElementById("fraud-counter");
    
    fraudCounter.textContent = state.fraudAlerts;

    if (state.transactions.length === 0) {
        feed.innerHTML = `<div class="empty-state"><p>No transactions parsed yet.</p></div>`;
        return;
    }

    feed.innerHTML = state.transactions.map((txn, index) => {
        const statusClass = txn.isFraud ? "fraud" : "safe";
        const statusBadge = txn.isFraud ? "badge-fraud" : "badge-safe";
        const statusLabel = txn.isFraud ? "Blocked" : "Safe";
        const scorePct = Math.round(txn.fraudScore * 100);
        
        return `
            <div class="txn-card ${txn.isFraud ? 'fraud' : ''}" data-index="${index}">
                <div class="txn-left">
                    <div class="txn-icon ${statusClass}">
                        ${txn.payee.charAt(0).toUpperCase()}
                    </div>
                    <div class="txn-meta">
                        <div class="payee">${txn.payee}</div>
                        <div class="sub-info">${txn.time} · ${txn.category} · ${txn.app} (${txn.device})</div>
                    </div>
                </div>
                <div class="txn-right">
                    <div class="amount" style="color: ${txn.isFraud ? '#ef4444' : '#f8fafc'}">
                        ₹${txn.amount.toFixed(2)}
                    </div>
                    <span class="badge ${statusBadge}">${statusLabel} (${scorePct}%)</span>
                </div>
            </div>
        `;
    }).join("");
}

// ==========================================================================
//  SMS Interceptor Sandbox
// ==========================================================================
function initTestingControls() {
    const btnSendSms = document.getElementById("btn-send-sms");
    const smsBodyInput = document.getElementById("test-sms-body");

    btnSendSms.addEventListener("click", async () => {
        const body = smsBodyInput.value.trim();
        if (!body) return;

        // Perform client-side Layer 1 parsing mimicking kotlin receiver Gates
        // Gate 1: Check keywords for transactions
        const keywords = ["debited", "credited", "spent", "alert", "withdrawn"];
        const matchesKeyword = keywords.some(k => body.toLowerCase().includes(k));
        
        if (!matchesKeyword) {
            alert("Layer 1 Interceptor Rejected: No transaction keyword matched.");
            return;
        }

        // Gate 2: Extract details via regular expressions
        // Amount matcher (looks for Rs., INR, ₹ followed by number)
        const amtRegex = /(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{2})?)/i;
        const amtMatch = body.match(amtRegex);
        const amount = amtMatch ? parseFloat(amtMatch[1].replace(/,/g, "")) : 100.0;

        // Payee matcher (looks for 'to ...' or 'towards ...' or 'at ...')
        const payeeRegex = /(?:to|towards|at|debited to)\s+([A-Za-z0-9\s]+?)(?:\s+(?:UPI|Ref|on|Bal|a\/c|from|$))/i;
        const payeeMatch = body.match(payeeRegex);
        const payee = payeeMatch ? payeeMatch[1].trim() : "Unknown Merchant";

        // Category Classifier (simple local rules)
        let category = "Uncategorized";
        const payeeLower = payee.toLowerCase();
        if (payeeLower.includes("zomato") || payeeLower.includes("swiggy") || payeeLower.includes("starbucks") || payeeLower.includes("hotel")) {
            category = "Food & Dining";
        } else if (payeeLower.includes("amazon") || payeeLower.includes("flipkart") || payeeLower.includes("myntra")) {
            category = "Shopping";
        } else if (payeeLower.includes("grocery") || payeeLower.includes("supermarket") || payeeLower.includes("mart")) {
            category = "Grocery";
        } else if (payeeLower.includes("uber") || payeeLower.includes("ola") || payeeLower.includes("irctc")) {
            category = "Travel";
        } else if (payeeLower.includes("netflix") || payeeLower.includes("spotify") || payeeLower.includes("multiplex")) {
            category = "Entertainment";
        }

        alert(`Layer 1 Intercept OK!\nParsed: Payee="${payee}", Amount=₹${amount}, Category="${category}"\nSending to XGBoost Ensemble...`);
        
        await scoreAndSaveTransaction({
            payee,
            amount,
            category,
            app: "GPay",
            device: "Android",
            hour: new Date().getHours(),
            newDevice: 0,
            ipMismatch: 0
        });
    });
}

// ==========================================================================
//  Finance Tracker Tab render
// ==========================================================================
function renderFinance() {
    const container = document.getElementById("category-bars");
    const progress = document.getElementById("goal-progress");
    const percentageText = document.getElementById("goal-percentage");

    // Aggregate category spends
    const categoryTotals = {};
    let total = 0;
    
    state.transactions.forEach(t => {
        if (!t.isFraud) {
            categoryTotals[t.category] = (categoryTotals[t.category] || 0) + t.amount;
            total += t.amount;
        }
    });

    state.monthlySpent = total;

    // Render bars
    const categories = ["Food & Dining", "Shopping", "Grocery", "Travel", "Entertainment", "Uncategorized"];
    
    container.innerHTML = categories.map(cat => {
        const spent = categoryTotals[cat] || 0;
        const percent = total > 0 ? (spent / total) * 100 : 0;
        
        let color = "var(--brand-primary)";
        if (cat === "Food & Dining") color = "var(--brand-purple)";
        if (cat === "Shopping") color = "var(--brand-warning)";
        
        return `
            <div class="cat-bar-item">
                <div class="cat-bar-info">
                    <span class="cat-name">${cat}</span>
                    <span class="cat-percent">₹${spent.toFixed(2)} (${percent.toFixed(0)}%)</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: ${percent}%; background-color: ${color};"></div>
                </div>
            </div>
        `;
    }).join("");

    // Render goals progress
    const goalPercent = Math.min(100, (state.monthlySpent / state.budgetLimit) * 100);
    progress.style.width = `${goalPercent}%`;
    percentageText.textContent = `${goalPercent.toFixed(0)}% spent (₹${state.monthlySpent.toFixed(2)} / ₹${state.budgetLimit.toFixed(2)})`;

    if (goalPercent > 90) {
        progress.style.backgroundColor = "var(--brand-danger)";
    } else if (goalPercent > 70) {
        progress.style.backgroundColor = "var(--brand-warning)";
    } else {
        progress.style.backgroundColor = "var(--brand-success)";
    }
}

// ==========================================================================
//  AI Guardian Chatbot Assistant
// ==========================================================================
function initChatbot() {
    const input = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send-btn");
    const box = document.getElementById("chat-box");
    const shortcuts = document.querySelectorAll(".chat-shortcut");

    const sendMessage = async (text) => {
        if (!text) return;
        
        // Add User Message
        appendMessage("user", text);
        input.value = "";
        
        // Show Typing Indicator
        const typingId = showTypingIndicator();
        
        // Get Response
        const reply = await generateAIResponse(text);
        
        // Remove Typing Indicator & Add reply
        removeTypingIndicator(typingId);
        appendMessage("assistant", reply);
    };

    sendBtn.addEventListener("click", () => sendMessage(input.value.trim()));
    
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendMessage(input.value.trim());
    });

    shortcuts.forEach(btn => {
        btn.addEventListener("click", () => {
            const query = btn.getAttribute("data-query");
            sendMessage(query);
        });
    });
}

function appendMessage(sender, text) {
    const box = document.getElementById("chat-box");
    const wrapper = document.createElement("div");
    wrapper.className = `message ${sender}`;
    wrapper.innerHTML = `<div class="message-bubble">${text}</div>`;
    box.appendChild(wrapper);
    box.scrollTop = box.scrollHeight;
}

function showTypingIndicator() {
    const box = document.getElementById("chat-box");
    const wrapper = document.createElement("div");
    const id = "typing-" + Date.now();
    wrapper.id = id;
    wrapper.className = "message assistant";
    wrapper.innerHTML = `
        <div class="message-bubble typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    box.appendChild(wrapper);
    box.scrollTop = box.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

async function generateAIResponse(query) {
    const lowerQuery = query.toLowerCase();

    // 1. Spending Summary response (Dynamic stats check)
    if (lowerQuery.includes("summary") || lowerQuery.includes("spend")) {
        const topCategory = getTopSpendingCategory();
        return `📊 **Nishika's Spending Summary:**\n\n` + 
               `- **Total Outflow:** ₹${state.monthlySpent.toFixed(2)}\n` +
               `- **Top Category:** ${topCategory.name} (₹${topCategory.spent.toFixed(2)})\n` +
               `- **Fraud Alerts Blocked:** ${state.fraudAlerts} suspicious attempts\n` +
               `- **Budget Limit:** ₹${state.budgetLimit.toFixed(2)}\n\n` +
               `Your budget utilization is currently **${((state.monthlySpent / state.budgetLimit) * 100).toFixed(0)}%**. ` +
               `${state.monthlySpent > state.budgetLimit ? "⚠️ You have exceeded your weekly target!" : "✓ You are safely within your target limit."}`;
    }

    // 2. Savings Tips or recommendations (Calls GET /insights/weekly endpoint on backend!)
    if (lowerQuery.includes("save") || lowerQuery.includes("tip") || lowerQuery.includes("dining")) {
        const topCategory = getTopSpendingCategory();
        
        try {
            // Fetch insights dynamically from the FastAPI server!
            const params = new URLSearchParams({
                total_spent: state.monthlySpent,
                top_category: topCategory.name,
                top_category_pct: topCategory.pct,
                fraud_alerts: state.fraudAlerts,
                vs_last_week_pct: 12.0 // Mock change pace
            });
            
            const response = await fetch(`${CONFIG.apiBase}/insights/weekly?${params.toString()}`, {
                method: "GET",
                headers: { "Authorization": `Bearer ${state.token}` }
            });
            
            if (response.ok) {
                const data = await response.json();
                return `💡 **AI Savings Recommendation:**\n\n` +
                       `"${data.savings_tip}"\n\n` +
                       `*Budget Pace: ${data.budget_status}*`;
            }
        } catch (err) {
            console.warn("Backend Insights API offline, using rule fallback.");
        }

        // Local fallback if API call fails
        return `💡 **Local Saving Tip:**\n\nLimit restaurant visits to weekends. Cooking 2 extra meals at home saves ≈₹800/week on average.`;
    }

    // 3. Fraud Security Status
    if (lowerQuery.includes("fraud") || lowerQuery.includes("security") || lowerQuery.includes("alert")) {
        return `🛡️ **PaySense Security Status:**\n\n` +
               `- **Active Engine:** XGBoost 41-feature Ensemble & TRAI Sender Rule Engine.\n` +
               `- **Blocked Fraud Incidents:** ${state.fraudAlerts} hits.\n` +
               `- **Current Risk:** Low. Any transaction scoring above 0.70 will trigger immediate lockups and notification.`;
    }

    // 4. Default Greeting / General Q&A
    return `I can analyze your spending history and coordinate with your XGBoost model to provide savings tips. Try asking me:\n\n` +
           `1. "Give me a quick spending summary"\n` +
           `2. "How can I save money?"\n` +
           `3. "Tell me about my fraud alerts"`;
}

function getTopSpendingCategory() {
    const categoryTotals = {};
    let total = 0;
    
    state.transactions.forEach(t => {
        if (!t.isFraud) {
            categoryTotals[t.category] = (categoryTotals[t.category] || 0) + t.amount;
            total += t.amount;
        }
    });

    let topCatName = "Uncategorized";
    let topCatSpent = 0;

    for (const [cat, spent] of Object.entries(categoryTotals)) {
        if (spent > topCatSpent) {
            topCatName = cat;
            topCatSpent = spent;
        }
    }

    const pct = total > 0 ? (topCatSpent / total) * 100 : 0;
    return { name: topCatName, spent: topCatSpent, pct: pct };
}
