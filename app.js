/**
 * Application Controller & Canvas Visualizations
 * Project: 31. ABISHEK M - Medical Appointment No-Show Prediction
 * 
 * Manages:
 * - Tab navigation
 * - Real-time client-side ML predictions
 * - Dynamic Benchmark Table & Confusion Matrix rendering
 * - Canvas charts for ROC Curves, PR Curves, Lead Time & Feature Importances
 * - Batch appointment triage & CSV download
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initPredictor();
    initBenchmarkTable();
    initBatchTriage();
    drawAllCharts();
});

// ----------------- TAB NAVIGATION -----------------
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add('active');
                // Redraw canvas charts if hidden previously
                setTimeout(drawAllCharts, 50);
            }
        });
    });
}

// ----------------- REAL-TIME PATIENT PREDICTOR -----------------
function initPredictor() {
    const ageInput = document.getElementById('input-age');
    const ageVal = document.getElementById('val-age');
    const leadInput = document.getElementById('input-lead-days');
    const leadVal = document.getElementById('val-lead-days');
    const predictBtn = document.getElementById('btn-predict');

    ageInput.addEventListener('input', () => {
        ageVal.textContent = ageInput.value;
        runSinglePrediction();
    });

    leadInput.addEventListener('input', () => {
        leadVal.textContent = `${leadInput.value}d`;
        runSinglePrediction();
    });

    const triggerInputs = [
        'input-gender', 'input-scholarship', 'input-hypertension',
        'input-diabetes', 'input-alcoholism', 'input-handicap',
        'input-weekday', 'input-sms', 'input-prev-appts',
        'input-prev-noshows', 'input-model'
    ];

    triggerInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', runSinglePrediction);
            el.addEventListener('input', runSinglePrediction);
        }
    });

    predictBtn.addEventListener('click', (e) => {
        e.preventDefault();
        runSinglePrediction();
    });

    // Run initial prediction on page load
    runSinglePrediction();
}

function runSinglePrediction() {
    if (!window.mlEngine) return;

    const payload = {
        Age: Number(document.getElementById('input-age').value),
        Gender: document.getElementById('input-gender').value,
        Scholarship: document.getElementById('input-scholarship').checked ? 1 : 0,
        Hypertension: document.getElementById('input-hypertension').checked ? 1 : 0,
        Diabetes: document.getElementById('input-diabetes').checked ? 1 : 0,
        Alcoholism: document.getElementById('input-alcoholism').checked ? 1 : 0,
        Handicap: document.getElementById('input-handicap').checked ? 1 : 0,
        LeadDays: Number(document.getElementById('input-lead-days').value),
        Appt_DayOfWeek: Number(document.getElementById('input-weekday').value),
        SMS_received: Number(document.getElementById('input-sms').value),
        Prev_Appointments: Number(document.getElementById('input-prev-appts').value),
        Prev_NoShows: Number(document.getElementById('input-prev-noshows').value)
    };

    const modelName = document.getElementById('input-model').value;
    const res = window.mlEngine.predict(payload, modelName, 0.50);

    // Update UI elements
    const riskBox = document.getElementById('risk-box');
    const riskScore = document.getElementById('risk-score');
    const riskBadge = document.getElementById('risk-badge');
    const verdict = document.getElementById('prediction-verdict');
    const driverList = document.getElementById('driver-list');
    const recList = document.getElementById('rec-list');

    riskBox.style.borderColor = res.riskColor;
    riskScore.textContent = `${res.probabilityPercent}%`;
    riskScore.style.color = res.riskColor;

    riskBadge.textContent = res.riskTier;
    riskBadge.style.color = res.riskColor;
    riskBadge.style.backgroundColor = `${res.riskColor}22`;
    riskBadge.style.borderColor = `${res.riskColor}55`;

    verdict.textContent = res.predictionText;

    driverList.innerHTML = res.drivers.map(d => `<li>🔸 ${d}</li>`).join('');
    recList.innerHTML = res.recommendations.map(r => `<li>🎯 ${r}</li>`).join('');
}

// ----------------- MODEL BENCHMARK TABLE -----------------
function initBenchmarkTable() {
    const tbody = document.querySelector('#benchmark-table tbody');
    if (!tbody || !window.MODEL_DATA || !window.MODEL_DATA.benchmarks) return;

    const benchmarks = window.MODEL_DATA.benchmarks;
    const rowsHtml = Object.keys(benchmarks).map(modelName => {
        const m = benchmarks[modelName];
        const isXGB = modelName === 'XGBoost';
        return `
            <tr style="${isXGB ? 'background: rgba(6, 182, 212, 0.08); font-weight: bold;' : ''}">
                <td style="color: ${isXGB ? '#06B6D4' : '#F8FAFC'};">
                    ${isXGB ? '★ ' : ''}${modelName}
                </td>
                <td>${(m.accuracy * 100).toFixed(2)}%</td>
                <td style="color: #34D399;">${(m.recall * 100).toFixed(2)}%</td>
                <td>${(m.precision * 100).toFixed(2)}%</td>
                <td>${(m.specificity * 100).toFixed(2)}%</td>
                <td>${m.f1_score.toFixed(4)}</td>
                <td style="color: #60A5FA; font-weight: bold;">${m.roc_auc.toFixed(4)}</td>
                <td style="color: #F472B6; font-weight: bold;">${m.pr_auc.toFixed(4)}</td>
            </tr>
        `;
    }).join('');

    tbody.innerHTML = rowsHtml;
}

// ----------------- BATCH SCHEDULING TRIAGE -----------------
let batchScoredData = [];

function initBatchTriage() {
    const tbody = document.getElementById('batch-tbody');
    const exportBtn = document.getElementById('btn-export-csv');
    if (!tbody || !window.MODEL_DATA || !window.MODEL_DATA.sample_schedule) return;

    const samples = window.MODEL_DATA.sample_schedule;
    batchScoredData = samples.map((row, idx) => {
        const res = window.mlEngine.predict(row, "XGBoost", 0.50);
        return {
            id: row.AppointmentID || (5600000 + idx),
            age: row.Age,
            gender: row.Gender || (row.Gender_F ? 'F' : 'M'),
            leadDays: row.LeadDays !== undefined ? row.LeadDays : 5,
            sms: row.SMS_received ? 'Yes' : 'No',
            prevNoShows: row.Prev_NoShows || 0,
            prob: res.probabilityPercent,
            probDec: res.probability,
            tier: res.riskTier,
            tierColor: res.riskColor,
            rec: res.recommendations[0]
        };
    });

    // Sort descending by risk probability
    batchScoredData.sort((a, b) => b.prob - a.prob);

    // Update KPI counters
    const high = batchScoredData.filter(d => d.tier === 'High Risk').length;
    const mod = batchScoredData.filter(d => d.tier === 'Moderate Risk').length;
    const low = batchScoredData.filter(d => d.tier === 'Low Risk').length;

    document.getElementById('batch-total').textContent = batchScoredData.length;
    document.getElementById('batch-high').textContent = high;
    document.getElementById('batch-mod').textContent = mod;
    document.getElementById('batch-low').textContent = low;

    // Render table rows
    tbody.innerHTML = batchScoredData.map(d => `
        <tr>
            <td style="font-family: monospace; color: #94A3B8;">#${d.id}</td>
            <td>${d.age} yo (${d.gender})</td>
            <td>${d.leadDays} days</td>
            <td>${d.sms}</td>
            <td>${d.prevNoShows} missed</td>
            <td style="font-weight: bold; color: ${d.tierColor};">${d.prob}%</td>
            <td>
                <span style="background: ${d.tierColor}22; color: ${d.tierColor}; border: 1px solid ${d.tierColor}44; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700;">
                    ${d.tier}
                </span>
            </td>
            <td style="font-size: 0.8rem; color: #CBD5E1;">${d.rec}</td>
        </tr>
    `).join('');

    // CSV Download
    exportBtn.addEventListener('click', () => {
        let csvContent = "data:text/csv;charset=utf-8,AppointmentID,Age,Gender,LeadDays,SMS_Received,Prev_NoShows,NoShow_Risk,Risk_Tier,Recommendation\n";
        batchScoredData.forEach(d => {
            csvContent += `"${d.id}","${d.age}","${d.gender}","${d.leadDays}","${d.sms}","${d.prevNoShows}","${d.prob}%","${d.tier}","${d.rec}"\n`;
        });
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "patient_outreach_priorities.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}

// ----------------- HIGH-PERFORMANCE CANVAS CHARTS -----------------
function drawAllCharts() {
    drawROCChart();
    drawPRChart();
    drawLeadTimeChart();
    drawAgeChart();
    drawFeatureImportanceChart();
}

function drawROCChart() {
    const canvas = document.getElementById('canvas-roc');
    if (!canvas || !window.MODEL_DATA) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Padding
    const padL = 50, padR = 20, padT = 20, padB = 40;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Grid & Axes
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    // Random guess diagonal
    ctx.strokeStyle = '#64748B';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, h - padB);
    ctx.lineTo(w - padR, padT);
    ctx.stroke();
    ctx.setLineDash([]);

    // Curve data
    const curves = window.MODEL_DATA.roc_pr_curves;
    const colors = {
        'XGBoost': '#06B6D4',
        'Random Forest': '#EC4899',
        'Decision Tree': '#8B5CF6',
        'Logistic Regression': '#3B82F6'
    };

    Object.keys(colors).forEach(model => {
        if (!curves[model]) return;
        const ptsFpr = curves[model].fpr;
        const ptsTpr = curves[model].tpr;

        ctx.strokeStyle = colors[model];
        ctx.lineWidth = model === 'XGBoost' ? 2.8 : 1.8;
        ctx.beginPath();

        for (let i = 0; i < ptsFpr.length; i++) {
            const x = padL + ptsFpr[i] * plotW;
            const y = (h - padB) - ptsTpr[i] * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    });

    // Axis Labels
    ctx.fillStyle = '#94A3B8';
    ctx.font = '11px "Plus Jakarta Sans", sans-serif';
    ctx.fillText('False Positive Rate (FPR)', padL + plotW / 2 - 60, h - 10);
    ctx.save();
    ctx.translate(16, padT + plotH / 2 + 50);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('True Positive Rate (Recall)', 0, 0);
    ctx.restore();
}

function drawPRChart() {
    const canvas = document.getElementById('canvas-pr');
    if (!canvas || !window.MODEL_DATA) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 50, padR = 20, padT = 20, padB = 40;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    // Baseline horizontal line at 20.2%
    const baseY = (h - padB) - 0.2019 * plotH;
    ctx.strokeStyle = '#64748B';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, baseY);
    ctx.lineTo(w - padR, baseY);
    ctx.stroke();
    ctx.setLineDash([]);

    const curves = window.MODEL_DATA.roc_pr_curves;
    const colors = {
        'XGBoost': '#06B6D4',
        'Random Forest': '#EC4899'
    };

    Object.keys(colors).forEach(model => {
        if (!curves[model]) return;
        const ptsRec = curves[model].recall;
        const ptsPrec = curves[model].precision;

        ctx.strokeStyle = colors[model];
        ctx.lineWidth = model === 'XGBoost' ? 2.8 : 2.0;
        ctx.beginPath();

        for (let i = 0; i < ptsRec.length; i++) {
            const x = padL + ptsRec[i] * plotW;
            const y = (h - padB) - ptsPrec[i] * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    });

    ctx.fillStyle = '#94A3B8';
    ctx.font = '11px "Plus Jakarta Sans", sans-serif';
    ctx.fillText('Recall (Sensitivity)', padL + plotW / 2 - 50, h - 10);
    ctx.save();
    ctx.translate(16, padT + plotH / 2 + 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Precision (PPV)', 0, 0);
    ctx.restore();
}

function drawLeadTimeChart() {
    const canvas = document.getElementById('canvas-lead-trend');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 45, padR = 20, padT = 20, padB = 35;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Data points: lead days vs no-show rate
    const points = [
        { d: 0, rate: 0.046 },
        { d: 2, rate: 0.170 },
        { d: 5, rate: 0.235 },
        { d: 10, rate: 0.278 },
        { d: 15, rate: 0.315 },
        { d: 20, rate: 0.332 },
        { d: 30, rate: 0.345 },
        { d: 45, rate: 0.358 }
    ];

    ctx.strokeStyle = '#334155';
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, h - padB);
    ctx.lineTo(w - padR, h - padB);
    ctx.stroke();

    // Area fill
    ctx.beginPath();
    points.forEach((pt, i) => {
        const x = padL + (pt.d / 45) * plotW;
        const y = (h - padB) - (pt.rate / 0.40) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.lineTo(padL + plotW, h - padB);
    ctx.lineTo(padL, h - padB);
    ctx.closePath();
    ctx.fillStyle = 'rgba(56, 189, 248, 0.15)';
    ctx.fill();

    // Line
    ctx.strokeStyle = '#38BDF8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    points.forEach((pt, i) => {
        const x = padL + (pt.d / 45) * plotW;
        const y = (h - padB) - (pt.rate / 0.40) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots
    points.forEach(pt => {
        const x = padL + (pt.d / 45) * plotW;
        const y = (h - padB) - (pt.rate / 0.40) * plotH;
        ctx.fillStyle = '#0284C7';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
    });

    ctx.fillStyle = '#94A3B8';
    ctx.font = '10px "Plus Jakarta Sans", sans-serif';
    ctx.fillText('0d (Same-Day: 4.6%)', padL, h - 12);
    ctx.fillText('15d (31.5%)', padL + (15 / 45) * plotW - 20, h - 12);
    ctx.fillText('45d (35.8%)', padL + plotW - 60, h - 12);
}

function drawAgeChart() {
    const canvas = document.getElementById('canvas-age-trend');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 45, padR = 20, padT = 20, padB = 35;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const cohorts = [
        { age: 'Child (0-12)', rate: 0.198 },
        { age: 'Youth (13-24)', rate: 0.254 },
        { age: 'Adult (25-44)', rate: 0.218 },
        { age: 'Middle (45-64)', rate: 0.185 },
        { age: 'Senior (65+)', rate: 0.148 }
    ];

    const barW = plotW / cohorts.length - 16;
    cohorts.forEach((c, idx) => {
        const x = padL + idx * (plotW / cohorts.length) + 8;
        const barH = (c.rate / 0.30) * plotH;
        const y = (h - padB) - barH;

        ctx.fillStyle = c.rate > 0.22 ? '#F43F5E' : '#3B82F6';
        ctx.fillRect(x, y, barW, barH);

        // Labels
        ctx.fillStyle = '#E2E8F0';
        ctx.font = '10px "Plus Jakarta Sans", sans-serif';
        ctx.fillText(`${(c.rate * 100).toFixed(1)}%`, x + 4, y - 6);

        ctx.fillStyle = '#94A3B8';
        ctx.fillText(c.age.split(' ')[0], x, h - 12);
    });
}

function drawFeatureImportanceChart() {
    const canvas = document.getElementById('canvas-features');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const feats = [
        { name: 'SameDay_Appt', val: 0.848 },
        { name: 'LeadDays', val: 0.312 },
        { name: 'Prev_NoShows', val: 0.278 },
        { name: 'Prev_NoShow_Rate', val: 0.185 },
        { name: 'Patient Age', val: 0.125 },
        { name: 'SMS_received', val: 0.085 },
        { name: 'Scheduled_Hour', val: 0.045 }
    ];

    const padL = 120, padR = 40, padT = 20, padB = 20;
    const plotW = w - padL - padR;
    const rowH = (h - padT - padB) / feats.length;

    feats.forEach((f, idx) => {
        const y = padT + idx * rowH;
        const barW = f.val * plotW;

        ctx.fillStyle = '#CBD5E1';
        ctx.font = '11px "Plus Jakarta Sans", sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(f.name, padL - 10, y + 15);

        ctx.fillStyle = idx === 0 ? '#06B6D4' : '#3B82F6';
        ctx.fillRect(padL, y + 4, barW, 14);

        ctx.textAlign = 'left';
        ctx.fillStyle = '#94A3B8';
        ctx.fillText(f.val.toFixed(3), padL + barW + 6, y + 15);
    });
}
