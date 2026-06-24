/**
 * index.js — Solar Flare Dashboard Client
 * 
 * Real-time visualization of X-ray light curves from Aditya-L1,
 * flare detection alerts, and forecast probability gauge.
 * 
 * Connects via WebSocket to the FastAPI backend for live data streaming.
 */

// ===================================================================
// CONFIG
// ===================================================================
const WS_URL = `ws://${window.location.host}/ws/stream`;
const API_BASE = window.location.origin;
const MAX_CHART_POINTS = 2000;     // max data points on chart
const ALERT_MAX_DISPLAY = 20;      // max alerts to show

// ===================================================================
// STATE
// ===================================================================
let ws = null;
let chart = null;
let chartView = 'normalized';      // 'normalized' or 'raw'
let streamSpeed = 50;
let alerts = [];
let totalPoints = 0;
let currentPosition = 0;

// Chart data buffers
const chartData = {
    labels: [],
    soft: [],
    hard: [],
    softRaw: [],
    hardRaw: [],
};

// ===================================================================
// INITIALIZATION
// ===================================================================

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    initClock();
    initSpeedControl();
    loadInitialData();
    connectWebSocket();
});

// ===================================================================
// CHART.JS — Dual-axis light curve chart
// ===================================================================

function initChart() {
    const ctx = document.getElementById('lightCurveChart').getContext('2d');
    
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Soft X-ray (SoLEXS)',
                    data: [],
                    borderColor: '#FFB347',
                    backgroundColor: 'rgba(255,179,71,0.05)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3,
                    yAxisID: 'y',
                },
                {
                    label: 'Hard X-ray (HEL1OS)',
                    data: [],
                    borderColor: '#00E5FF',
                    backgroundColor: 'rgba(0,229,255,0.03)',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3,
                    yAxisID: 'y',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0,
            },
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#8892A8',
                        font: { family: 'Inter', size: 11 },
                        padding: 16,
                        usePointStyle: true,
                        pointStyle: 'line',
                    },
                },
                tooltip: {
                    backgroundColor: 'rgba(13,18,37,0.95)',
                    titleColor: '#E8ECF4',
                    bodyColor: '#8892A8',
                    borderColor: 'rgba(255,179,71,0.2)',
                    borderWidth: 1,
                    titleFont: { family: 'JetBrains Mono', size: 11 },
                    bodyFont: { family: 'JetBrains Mono', size: 11 },
                    padding: 10,
                    cornerRadius: 8,
                },
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time (seconds)',
                        color: '#5A6380',
                        font: { family: 'Inter', size: 11 },
                    },
                    ticks: {
                        color: '#5A6380',
                        font: { family: 'JetBrains Mono', size: 10 },
                        maxTicksLimit: 10,
                        callback: function(val) {
                            const v = this.getLabelForValue(val);
                            return formatTime(v);
                        },
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.03)',
                    },
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: chartView === 'normalized' ? 'Normalized Flux (σ)' : 'Count Rate',
                        color: '#5A6380',
                        font: { family: 'Inter', size: 11 },
                    },
                    ticks: {
                        color: '#5A6380',
                        font: { family: 'JetBrains Mono', size: 10 },
                    },
                    grid: {
                        color: 'rgba(255,255,255,0.03)',
                    },
                },
            },
        },
    });
}

function setChartView(view) {
    chartView = view;
    document.querySelectorAll('.chart-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === view);
    });
    
    // Update axis label
    chart.options.scales.y.title.text = 
        view === 'normalized' ? 'Normalized Flux (σ)' : 'Count Rate';
    
    updateChartData();
}

function updateChartData() {
    if (!chart) return;
    
    chart.data.labels = chartData.labels;
    
    if (chartView === 'normalized') {
        chart.data.datasets[0].data = chartData.soft;
        chart.data.datasets[1].data = chartData.hard;
    } else {
        chart.data.datasets[0].data = chartData.softRaw;
        chart.data.datasets[1].data = chartData.hardRaw;
    }
    
    chart.update('none');
}

function addDataToChart(points) {
    for (const p of points) {
        chartData.labels.push(p.time_s);
        chartData.soft.push(p.soft_norm);
        chartData.hard.push(p.hard_norm);
        chartData.softRaw.push(p.soft);
        chartData.hardRaw.push(p.hard);
    }
    
    // Trim to max points
    while (chartData.labels.length > MAX_CHART_POINTS) {
        chartData.labels.shift();
        chartData.soft.shift();
        chartData.hard.shift();
        chartData.softRaw.shift();
        chartData.hardRaw.shift();
    }
    
    updateChartData();
}

// ===================================================================
// FORECAST GAUGE
// ===================================================================

function updateGauge(probability, isAlert) {
    const gaugeValue = document.getElementById('gaugeValue');
    const gaugeFill = document.getElementById('gaugeFill');
    
    const pct = Math.min(Math.max(probability * 100, 0), 100);
    const arcLength = 236; // total arc length of the SVG path
    const offset = arcLength - (pct / 100) * arcLength;
    
    gaugeFill.style.strokeDashoffset = offset;
    gaugeValue.textContent = `${pct.toFixed(0)}%`;
    
    // Color based on probability
    let color;
    if (pct < 30) {
        color = '#4CAF50'; // green
        gaugeFill.style.stroke = color;
    } else if (pct < 60) {
        color = '#FF9800'; // orange
        gaugeFill.style.stroke = color;
    } else {
        color = '#F44336'; // red
        gaugeFill.style.stroke = color;
    }
    
    gaugeValue.style.color = color;
    
    // Pulse the solar logo on high probability
    const logo = document.getElementById('solarLogo');
    if (pct > 70) {
        logo.style.animation = 'pulse-glow 0.8s ease-in-out infinite';
    } else {
        logo.style.animation = 'pulse-glow 3s ease-in-out infinite';
    }
}

// ===================================================================
// ALERTS
// ===================================================================

function addAlert(alertData) {
    alerts.unshift(alertData);
    if (alerts.length > ALERT_MAX_DISPLAY) alerts.pop();
    
    renderAlerts();
    
    // Update alert count badge
    document.getElementById('alertCount').textContent = alerts.length;
    
    // Flash the status indicator
    const statusDot = document.getElementById('statusDot');
    statusDot.className = 'status-dot danger';
    setTimeout(() => { statusDot.className = 'status-dot'; }, 3000);
}

function renderAlerts() {
    const container = document.getElementById('alertList');
    
    if (alerts.length === 0) {
        container.innerHTML = `
            <div style="text-align:center;color:var(--text-muted);padding:24px;font-size:0.8rem;">
                No flare alerts yet
            </div>`;
        return;
    }
    
    container.innerHTML = alerts.map(a => `
        <div class="alert-item" style="border-color:${a.color || 'rgba(255,87,34,0.2)'};">
            <div style="display:flex;align-items:center;">
                <span class="alert-class-badge alert-class-${a.flare_class}">${a.flare_class}</span>
                <div class="alert-details">
                    <span style="font-weight:600;color:var(--text-primary);">
                        ${a.flare_class}-class Flare Detected
                    </span>
                    <span class="alert-time">t = ${formatTime(a.peak_time)}</span>
                    <span class="alert-type type-badge type-${a.detection_type}">
                        ${a.detection_type.replace('_', ' ')}
                    </span>
                </div>
            </div>
            <div style="margin-top:6px;font-size:0.7rem;color:var(--text-muted);">
                Peak: ${a.peak_soft?.toFixed(1)}σ · Conf: ${(a.confidence * 100).toFixed(0)}% · Dur: ${a.duration?.toFixed(0)}s
            </div>
        </div>
    `).join('');
}

// ===================================================================
// CATALOGUE TABLE
// ===================================================================

function loadCatalogue(events) {
    const tbody = document.getElementById('catalogueBody');
    const count = document.getElementById('catalogueCount');
    
    if (!events || events.length === 0) {
        tbody.innerHTML = `
            <tr><td colspan="10" style="text-align:center;color:var(--text-muted);padding:24px;">
                No events detected yet
            </td></tr>`;
        count.textContent = '0 events';
        return;
    }
    
    count.textContent = `${events.length} events`;
    
    tbody.innerHTML = events.map((e, i) => `
        <tr>
            <td>${e.event_id ?? i}</td>
            <td><span class="class-pill class-pill-${e.flare_class}">${e.flare_class}</span></td>
            <td>${e.start_time?.toFixed(0)}</td>
            <td>${e.peak_time?.toFixed(0)}</td>
            <td>${e.duration?.toFixed(0)}s</td>
            <td>${e.peak_soft?.toFixed(1)}σ</td>
            <td>${e.peak_hard?.toFixed(1)}σ</td>
            <td><span class="type-badge type-${e.detection_type}">${(e.detection_type || '').replace('_', ' ')}</span></td>
            <td>
                <div style="display:flex;align-items:center;gap:6px;">
                    <div class="conf-bar" style="width:${(e.confidence * 60)}px;background:${getConfColor(e.confidence)}"></div>
                    <span>${(e.confidence * 100).toFixed(0)}%</span>
                </div>
            </td>
            <td>${e.hard_lead_time ? e.hard_lead_time.toFixed(0) + 's' : '—'}</td>
        </tr>
    `).join('');
}

function getConfColor(conf) {
    if (conf > 0.8) return '#4CAF50';
    if (conf > 0.5) return '#FF9800';
    return '#FF5722';
}

// ===================================================================
// METRICS
// ===================================================================

function updateMetrics(metrics) {
    if (!metrics) return;
    
    const nc = metrics.nowcast || {};
    const fc = metrics.forecast || {};
    const lt = metrics.lead_time || {};
    
    setStatValue('statTPR', nc.tpr, 'info');
    setStatValue('statFAR', nc.far, nc.far < 0.1 ? 'good' : 'warning');
    setStatValue('statTSS', fc.tss, 'info');
    setStatValue('statAUC', fc.auc, '');
    setStatValue('statLeadTime', lt.mean_lead_min, 'warning', 'min');
    
    document.getElementById('statEvents').textContent = nc.tp ?? 0;
}

function setStatValue(id, value, colorClass, suffix = '') {
    const el = document.getElementById(id);
    if (!el || value === undefined || value === null) return;
    
    const formatted = typeof value === 'number' ? value.toFixed(2) : value;
    el.textContent = suffix ? `${formatted} ${suffix}` : formatted;
}

// ===================================================================
// WEBSOCKET
// ===================================================================

function connectWebSocket() {
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        document.getElementById('statusDot').className = 'status-dot';
        document.getElementById('statusText').textContent = 'Streaming';
        document.getElementById('loadingOverlay').classList.add('hidden');
        
        // Set initial speed
        ws.send(JSON.stringify({ action: 'set_speed', speed: streamSpeed }));
    };
    
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        
        switch (msg.type) {
            case 'data':
                if (msg.points) {
                    addDataToChart(msg.points);
                    currentPosition = msg.position || 0;
                    totalPoints = msg.total || 1;
                    
                    // Update progress bar
                    const pct = (currentPosition / totalPoints) * 100;
                    document.getElementById('progressBar').style.width = `${pct}%`;
                }
                break;
            
            case 'alert':
                addAlert(msg);
                break;
            
            case 'forecast':
                updateGauge(msg.probability, msg.alert);
                if (msg.reasons) {
                    updateReasoning(msg.reasons, msg.alert);
                }
                break;
            
            case 'error':
                console.error('Server error:', msg.message);
                break;
        }
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        document.getElementById('statusDot').className = 'status-dot warning';
        document.getElementById('statusText').textContent = 'Reconnecting...';
        
        // Auto-reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        document.getElementById('statusDot').className = 'status-dot danger';
        document.getElementById('statusText').textContent = 'Error';
    };
}

// ===================================================================
// INITIAL DATA LOAD
// ===================================================================

async function loadInitialData() {
    try {
        // Load catalogue
        const catRes = await fetch(`${API_BASE}/api/catalogue`);
        const catalogue = await catRes.json();
        loadCatalogue(catalogue);
        
        // Load metrics
        const metRes = await fetch(`${API_BASE}/api/metrics`);
        const metrics = await metRes.json();
        updateMetrics(metrics);
        
        // Load status
        const statusRes = await fetch(`${API_BASE}/api/status`);
        const status = await statusRes.json();
        
        if (status.forecast_horizon_min) {
            document.getElementById('horizonBadge').textContent = `${status.forecast_horizon_min} min`;
            document.getElementById('gaugeHorizon').textContent = status.forecast_horizon_min;
        }
        
        document.getElementById('statEvents').textContent = status.total_events || 0;
        
    } catch (err) {
        console.error('Failed to load initial data:', err);
    }
}

// ===================================================================
// CLOCK
// ===================================================================

function initClock() {
    function updateClock() {
        const now = new Date();
        const utc = now.toUTCString().split(' ')[4];
        document.getElementById('clockDisplay').textContent = `${utc} UTC`;
    }
    updateClock();
    setInterval(updateClock, 1000);
}

// ===================================================================
// SPEED CONTROL
// ===================================================================

function initSpeedControl() {
    const slider = document.getElementById('speedSlider');
    const label = document.getElementById('speedValue');
    
    slider.addEventListener('input', (e) => {
        streamSpeed = parseInt(e.target.value);
        label.textContent = `${streamSpeed}×`;
        
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'set_speed', speed: streamSpeed }));
        }
    });
}

function updateReasoning(reasons, isAlert) {
    const list = document.getElementById('aiReasoningList');
    if (!list) return;
    
    list.innerHTML = reasons.map(r => `
        <li style="display: flex; align-items: start; gap: 6px; line-height: 1.3;">
            <span style="color: ${isAlert ? 'var(--solar-orange)' : 'var(--class-a)'}; margin-top: 1px;">►</span>
            <span>${r}</span>
        </li>
    `).join('');
}

// ===================================================================
// UTILITIES
// ===================================================================

function formatTime(seconds) {
    if (!seconds && seconds !== 0) return '--:--';
    const totalSec = Math.floor(seconds);
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    
    if (h > 0) {
        return `${h}h${String(m).padStart(2, '0')}m`;
    }
    return `${m}:${String(s).padStart(2, '0')}`;
}
