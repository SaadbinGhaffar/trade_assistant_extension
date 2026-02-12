// content.js - Injects the dashboard into TradingView
console.log("Forex Assistant: Content script loaded.");

let panel = null;
let isDragging = false;
let dragOffsetX, dragOffsetY;

// --- Initialization ---
function init() {
    if (document.getElementById('fx-assistant-panel')) return;
    createPanel();
    fetchData(); // Initial fetch
}

// Watch for URL changes (SPA navigation)
let lastUrl = location.href;
new MutationObserver(() => {
    const url = location.href;
    if (url !== lastUrl) {
        lastUrl = url;
        fetchData();
    }
}).observe(document, { subtree: true, childList: true });

// Also re-inject if panel disappears (TV redraws)
setInterval(() => {
    if (!document.getElementById('fx-assistant-panel')) {
        createPanel();
    }
}, 3000);

// --- UI Creation ---
function createPanel() {
    panel = document.createElement('div');
    panel.id = 'fx-assistant-panel';
    panel.innerHTML = `
        <div id="fx-header">
            <span id="fx-title">FX Assistant</span>
            <span id="fx-close">×</span>
        </div>
        <div id="fx-content">
            <div id="fx-pair-info" class="fx-row">
                <span class="fx-label">Pair:</span>
                <span class="fx-value" id="val-pair">--</span>
            </div>
            <div class="fx-row">
                <span class="fx-label">Session:</span>
                <span class="fx-value" id="val-session">--</span>
            </div>
            
            <div class="score-container">
                <div class="fx-row">
                    <span class="fx-label">Long Score</span>
                    <span class="fx-value" id="val-long">0</span>
                </div>
                <div class="score-bar-bg"><div class="score-bar-fill score-long" id="bar-long"></div></div>
            </div>

            <div class="score-container">
                <div class="fx-row">
                    <span class="fx-label">Short Score</span>
                    <span class="fx-value" id="val-short">0</span>
                </div>
                <div class="score-bar-bg"><div class="score-bar-fill score-short" id="bar-short"></div></div>
            </div>

            <div class="fx-row">
                <span class="fx-label">Dominance:</span>
                <span class="fx-value" id="val-dom">0.0</span>
            </div>

            <hr style="border-color: #434651; opacity: 0.5;">

            <div class="fx-row">
                <span class="fx-label">Regime:</span>
                <span class="fx-value" id="val-regime">--</span>
            </div>
            <div class="fx-row">
                <span class="fx-label">Bias:</span>
                <span class="fx-value" id="val-bias">--</span>
            </div>
             <div class="fx-row">
                <span class="fx-label">Entry Quality:</span>
            <div class="fx-row">
                <span class="fx-label">Entry Quality:</span>
                <span class="fx-value" id="val-eq-score">--</span>
            </div>

            <div id="fx-status" class="status-blocked">Waiting...</div>
            
            <div id="fx-risk-panel" style="display:none; margin-top:10px; padding-top:10px; border-top:1px dashed #555;">
                 <div class="fx-row"><span class="fx-label" style="color:#fff">ENTRY PRICE:</span> <span class="fx-value" id="val-entry-price">--</span></div>
                 <div class="fx-row"><span class="fx-label">Stop Loss:</span> <span class="fx-value" id="val-stop">--</span></div>
                 <div class="fx-row"><span class="fx-label">Take Profit:</span> <span class="fx-value" id="val-target">--</span></div>
                 <div class="fx-row" style="margin-top:5px"><span class="fx-label">Lot Size:</span> <span class="fx-value" id="val-lot">--</span></div>
            </div>

            <button id="fx-refresh">Refresh Analysis</button>
        </div>
    `;

    document.body.appendChild(panel);

    // Event Listeners
    document.getElementById('fx-close').onclick = () => panel.style.display = 'none';
    document.getElementById('fx-refresh').onclick = fetchData;

    // Draggable logic
    const header = document.getElementById('fx-header');
    header.onmousedown = dragMouseDown;
}

function dragMouseDown(e) {
    e.preventDefault();
    dragOffsetX = e.clientX - panel.offsetLeft;
    dragOffsetY = e.clientY - panel.offsetTop;
    document.onmouseup = closeDragElement;
    document.onmousemove = elementDrag;
}

function elementDrag(e) {
    e.preventDefault();
    panel.style.top = (e.clientY - dragOffsetY) + "px";
    panel.style.left = (e.clientX - dragOffsetX) + "px";
}

function closeDragElement() {
    document.onmouseup = null;
    document.onmousemove = null;
}

// --- Logic ---
async function fetchData() {
    const btn = document.getElementById('fx-refresh');
    btn.disabled = true;
    btn.innerHTML = '<span class="fx-spinner"></span> Analyzing...';

    // 1. Get Pair from Page Title or URL
    // Better logic: Check if title contains known symbols
    let pair = "XAUUSD"; // Fallback
    const title = document.title.toUpperCase();

    // Priority check for supported pairs
    if (title.includes("EURUSD") || title.includes("EUR/USD")) pair = "EURUSD";
    else if (title.includes("XAGUSD") || title.includes("SILVER") || title.includes("XAG")) pair = "XAGUSD";
    else if (title.includes("GBPUSD") || title.includes("GBP/USD")) pair = "GBPUSD"; // Added support if config matches
    else if (title.includes("XAUUSD") || title.includes("GOLD") || title.includes("XAU")) pair = "XAUUSD";

    // Attempt to grab from URL if title failed (sometimes title is just "TradingView")
    const urlParams = new URLSearchParams(window.location.search);
    const symbolParam = urlParams.get('symbol');
    if (symbolParam) {
        if (symbolParam.includes("EURUSD")) pair = "EURUSD";
        if (symbolParam.includes("XAG")) pair = "XAGUSD";
        if (symbolParam.includes("XAU") || symbolParam.includes("GOLD")) pair = "XAUUSD";
    }

    try {
        const response = await fetch(`http://localhost:5000/analyze?pair=${pair}`);
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        render(data);
    } catch (err) {
        console.error("FX Assistant Error:", err);
        document.getElementById('fx-status').className = 'status-blocked';
        document.getElementById('fx-status').innerText = 'Connection Error';
    } finally {
        btn.disabled = false;
        btn.innerText = 'Refresh Analysis';
    }
}

function render(data) {
    document.getElementById('val-pair').innerText = data.pair;
    document.getElementById('val-session').innerText = `${data.session.name} (${data.session.time})`;

    // Scores
    const long = data.scores.long_total;
    const short = data.scores.short_total;

    document.getElementById('val-long').innerText = long.toFixed(1);
    document.getElementById('bar-long').style.width = long + '%';

    document.getElementById('val-short').innerText = short.toFixed(1);
    document.getElementById('bar-short').style.width = short + '%';

    document.getElementById('val-dom').innerText = data.scores.dominance_spread;

    // Details
    document.getElementById('val-regime').innerText =
        `${data.regime.type} (${(data.regime.confidence * 100).toFixed(0)}%)`;

    const db = data.scores.breakdown.daily_bias;
    const eq = data.scores.breakdown.entry_quality;
    if (db) document.getElementById('val-bias').innerText = `${db.details.label}`;
    if (eq) document.getElementById('val-eq-score').innerText = `${Math.max(eq.long, eq.short).toFixed(1)}/30`;

    // Status
    const statusBox = document.getElementById('fx-status');
    const riskPanel = document.getElementById('fx-risk-panel');

    if (data.eligibility.is_eligible) {
        statusBox.className = 'status-eligible';
        statusBox.innerText = `✅ ELIGIBLE ${data.eligibility.direction.toUpperCase()}`;
    } else {
        statusBox.className = 'status-blocked';
        const reason = data.eligibility.block_reasons[0] || "Unknown";
        statusBox.innerText = `🚫 BLOCKED: ${reason}`;
    }

    // Always show risk calc for visibility
    riskPanel.style.display = 'block';

    // Determine color based on direction/eligibility
    const isLong = data.scores.dominant_side === "Long";
    const color = isLong ? "#00bcd4" : "#ff5252";

    document.getElementById('val-lot').innerText = data.trade_setup.lot_size;
    document.getElementById('val-stop').innerHTML = `<span style="color:${color}">${data.trade_setup.stop}</span>`;
    document.getElementById('val-target').innerHTML = `<span style="color:${color}">${data.trade_setup.target}</span> (TP)`;
    document.getElementById('val-entry-price').innerHTML = `<span style="color:#fff; font-weight:bold">${data.trade_setup.entry}</span>`;
}

// Start
setTimeout(init, 2000); // Wait for TV to load
