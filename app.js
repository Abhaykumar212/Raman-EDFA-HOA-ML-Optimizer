/**
 * app.js
 * Main UI Orchestrator & State Controller (Neural Network Only version)
 * Binds all sliders, charts, canvas visualizers, and the ML optimization pipeline.
 */

// Helper to generate a random number within a range.
function randomVal(min, max) {
  return min + Math.random() * (max - min);
}

// Helper to sleep/yield execution to keep the UI responsive.
const sleepMs = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// State variables
let running = false;
let selectedBand = 'l-band'; // default L-band from paper

// Visualizer & Chart instances
let gainChart = null;
let nnLossChart = null;
let landscapeVis = null;
let nnVis = null;

/**
 * Update slider value displays dynamically.
 */
function updateSliderLabel(sliderId, labelId) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);
  if (slider && label) {
    label.textContent = slider.value;
  }
}

/**
 * Gets range values from the range sliders.
 */
function getParameterRanges() {
  return [
    [parseFloat(document.getElementById('s-rl-min').value), parseFloat(document.getElementById('s-rl-max').value)],
    [parseFloat(document.getElementById('s-el-min').value), parseFloat(document.getElementById('s-el-max').value)],
    [parseFloat(document.getElementById('s-rp-min').value), parseFloat(document.getElementById('s-rp-max').value)],
    [parseFloat(document.getElementById('s-ep-min').value), parseFloat(document.getElementById('s-ep-max').value)]
  ];
}

/**
 * Setup event listeners and label binds.
 */
function initializeSliders() {
  const binds = [
    { s: 's-rl-min', l: 'sv-rl-min' }, { s: 's-rl-max', l: 'sv-rl-max' },
    { s: 's-el-min', l: 'sv-el-min' }, { s: 's-el-max', l: 'sv-el-max' },
    { s: 's-rp-min', l: 'sv-rp-min' }, { s: 's-rp-max', l: 'sv-rp-max' },
    { s: 's-ep-min', l: 'sv-ep-min' }, { s: 's-ep-max', l: 'sv-ep-max' },
    { s: 'ml-samples', l: 'mv-samples' }, { s: 'ml-epochs', l: 'mv-epochs' },
    { s: 'ml-lr', l: 'mv-lr' }, { s: 'ml-steps', l: 'mv-steps' }
  ];

  binds.forEach(b => {
    const slider = document.getElementById(b.s);
    if (slider) {
      // Set initial label text
      updateSliderLabel(b.s, b.l);
      // Bind event
      slider.addEventListener('input', () => {
        updateSliderLabel(b.s, b.l);
        // If sliders that change landscape are updated, regenerate landscape cache
        if (b.s.startsWith('s-')) {
          triggerLandscapeUpdate();
        }
      });
    }
  });

  const bandSelect = document.getElementById('band-select');
  if (bandSelect) {
    bandSelect.addEventListener('change', (e) => {
      selectedBand = e.target.value;
      writeLog(`Wavelength band profile changed to ${selectedBand.toUpperCase()}`, 'system');
      triggerLandscapeUpdate();
    });
  }
}

/**
 * Regenerates the 2D contour landscape cache based on slider configurations.
 */
function triggerLandscapeUpdate() {
  if (landscapeVis) {
    const ranges = getParameterRanges();
    const mockParams = [
      (ranges[0][0] + ranges[0][1]) / 2, // rl
      (ranges[1][0] + ranges[1][1]) / 2, // el
      parseFloat(document.getElementById('s-rp-min').value), // rp
      parseFloat(document.getElementById('s-ep-min').value)  // ep
    ];
    
    // Evaluate function callback wrapper
    const evalWrapper = (ind) => {
      const g = generateWavelengthGrid(selectedBand, 50, selectedBand === 'l-band' ? 0.2 : 0.8);
      const gains = g.map(wl => calculateNetGain(wl, ind[0], ind[1], ind[2], ind[3], selectedBand));
      return calculateFitness(gains);
    };

    landscapeVis.ranges = ranges;
    landscapeVis.evaluateFn = evalWrapper;
    landscapeVis.generateLandscapeCache(mockParams, 30);
    landscapeVis.drawBackground();
  }
}

/**
 * Write log line inside the scrolling console.
 */
function writeLog(message, type = 'system') {
  const consoleEl = document.getElementById('log-box');
  if (consoleEl) {
    const line = document.createElement('div');
    line.className = `log-line log-line-${type}`;
    line.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
    consoleEl.appendChild(line);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }
}

/**
 * Set visual progress indicators in HTML.
 */
function updateProgress(percentage, text) {
  const fill = document.getElementById('prog-fill');
  const label = document.getElementById('prog-text');
  const val = document.getElementById('prog-pct');
  
  if (fill) fill.style.width = `${percentage}%`;
  if (label) label.textContent = text;
  if (val) val.textContent = `${percentage}%`;
}

/**
 * Refreshes values inside the metric widgets.
 */
function refreshMetrics(meanGain, flatness, noiseFigure, evaluations) {
  document.getElementById('m-gain').textContent = meanGain.toFixed(2);
  document.getElementById('m-flat').textContent = `±${(flatness / 2).toFixed(2)}`;
  document.getElementById('m-nf').textContent = noiseFigure.toFixed(2);
  document.getElementById('m-evals').textContent = evaluations;
}

/**
 * Destroys and initializes ChartJS objects.
 */
function initializeCharts() {
  // Gain Chart
  const ctxGain = document.getElementById('gainChart');
  if (gainChart) gainChart.destroy();
  
  gainChart = new Chart(ctxGain, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Ideal Target (20 dB)',
          data: [],
          borderColor: '#ef4444',
          borderWidth: 1.5,
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Wavelength (nm)', color: '#9CA3AF' },
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#9CA3AF' }
        },
        y: {
          title: { display: true, text: 'Net Gain (dB)', color: '#9CA3AF' },
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#9CA3AF' },
          min: 0,
          max: 40
        }
      }
    }
  });
}

/**
 * Updates dynamic datasets inside the charts.
 */
function updateChartDataset(chart, datasetLabel, dataPoints, strokeColor, isDash = false) {
  let dataset = chart.data.datasets.find(ds => ds.label === datasetLabel);
  
  if (!dataset) {
    dataset = {
      label: datasetLabel,
      data: [],
      borderColor: strokeColor,
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.3,
      borderDash: isDash ? [4, 4] : [],
      fill: false
    };
    chart.data.datasets.push(dataset);
  }
  
  dataset.data = dataPoints;
  chart.update('none');
}

/**
 * Display the best parameters inside the HTML parameters results box.
 */
function renderBestParameters(best, algoLabel) {
  const panel = document.getElementById('best-params');
  if (!panel || !best) return;
  
  const meanGain = best.gains.reduce((sum, g) => sum + g, 0) / best.gains.length;
  const flatness = Math.max(...best.gains) - Math.min(...best.gains);
  const nf = calculateNoiseFigure(best.rl, best.el, best.ep, selectedBand);

  panel.innerHTML = `
    <table class="results-table">
      <tr><td class="label">Optimizer Profile</td><td class="val" style="color: var(--accent-blue)">${algoLabel}</td></tr>
      <tr><td class="label">Raman Fiber Length (Lr)</td><td class="val">${best.rl.toFixed(2)} km</td></tr>
      <tr><td class="label">EDFA Active Length (Le)</td><td class="val">${best.el.toFixed(2)} m</td></tr>
      <tr><td class="label">Raman Pump Power (Pr)</td><td class="val">${best.rp.toFixed(0)} mW</td></tr>
      <tr><td class="label">EDFA Pump Power (Pe)</td><td class="val">${best.ep.toFixed(0)} mW</td></tr>
      <tr><td class="label">Optimal Net Gain</td><td class="val" style="color: var(--accent-green);">${meanGain.toFixed(2)} dB</td></tr>
      <tr><td class="label">Gain Flatness (Δ)</td><td class="val" style="color: var(--accent-amber);">±${(flatness / 2).toFixed(2)} dB</td></tr>
      <tr><td class="label">Noise Figure (NF)</td><td class="val" style="color: var(--accent-crimson);">${nf.toFixed(2)} dB</td></tr>
    </table>
  `;
}

/**
 * Runs the Neural Network Surrogate training and gradient ascent optimization.
 */
async function triggerML(ranges) {
  writeLog('Starting Neural Network Surrogate model optimization pipeline...', 'ml');
  
  const samplesCount = parseInt(document.getElementById('ml-samples').value);
  const epochs = parseInt(document.getElementById('ml-epochs').value);
  const lr = parseFloat(document.getElementById('ml-lr').value);
  const steps = parseInt(document.getElementById('ml-steps').value);
  
  const channels = selectedBand === 'l-band' ? 50 : 44;
  const spacing = selectedBand === 'l-band' ? 0.2 : 0.8;
  const wls = generateWavelengthGrid(selectedBand, channels, spacing);
  
  // Initialize NN architecture visualizer
  if (nnVis) {
    nnVis.drawNetwork(new SimpleNN(4, [16, 16], channels), 'idle');
  }

  // ── Step 1: Synthetic Dataset Generation ──
  writeLog(`Generating training dataset of ${samplesCount} samples...`, 'ml');
  updateProgress(5, 'ML: Generating dataset...');
  
  const dataset = [];
  for (let i = 0; i < samplesCount; i++) {
    const rl = randomVal(ranges[0][0], ranges[0][1]);
    const el = randomVal(ranges[1][0], ranges[1][1]);
    const rp = randomVal(ranges[2][0], ranges[2][1]);
    const ep = randomVal(ranges[3][0], ranges[3][1]);
    
    const gains = wls.map(wl => calculateNetGain(wl, rl, el, rp, ep, selectedBand));
    dataset.push({
      x: [rl, el, rp, ep],
      y: gains
    });
  }
  
  // Normalize dataset (inputs and outputs)
  const normData = dataset.map(sample => {
    const nx = sample.x.map((val, idx) => {
      const [min, max] = ranges[idx];
      return (val - min) / (max - min + 1e-9);
    });
    // scale output gains down by 40 to fit in tanh range nicely
    const ny = sample.y.map(g => g / 40.0);
    return { nx, ny };
  });

  // Setup Loss Chart
  const ctxLoss = document.getElementById('nnLossChart');
  if (nnLossChart) nnLossChart.destroy();
  const lossPoints = [];
  nnLossChart = new Chart(ctxLoss, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'MSE Loss',
        data: lossPoints,
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139,92,246,0.1)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9CA3AF', font: { size: 8 } } }
      }
    }
  });

  // ── Step 2: NN Training ──
  writeLog('Training Multi-Layer Perceptron (4x16x16x50) surrogate with Adam optimizer...', 'ml');
  const nn = new SimpleNN(4, [16, 16], channels);
  const batchSize = 16;
  
  for (let epoch = 1; epoch <= epochs; epoch++) {
    // Shuffle normData
    normData.sort(() => Math.random() - 0.5);
    
    let totalLoss = 0;
    
    for (let start = 0; start < normData.length; start += batchSize) {
      const end = Math.min(start + batchSize, normData.length);
      const batch = normData.slice(start, end);
      
      const accumulatedDW = [];
      const accumulatedDb = [];
      
      for (let l = 0; l < nn.layersCount - 1; l++) {
        accumulatedDW.push(Array.from({ length: nn.sizes[l] }, () => new Array(nn.sizes[l+1]).fill(0)));
        accumulatedDb.push(new Array(nn.sizes[l+1]).fill(0));
      }
      
      for (const sample of batch) {
        const pred = nn.forward(sample.nx);
        
        let sampleLoss = 0;
        for (let j = 0; j < pred.length; j++) {
          sampleLoss += Math.pow(pred[j] - sample.ny[j], 2);
        }
        totalLoss += sampleLoss / pred.length;
        
        const { dW, db } = nn.backward(sample.ny);
        for (let l = 0; l < nn.layersCount - 1; l++) {
          for (let j = 0; j < nn.sizes[l+1]; j++) {
            accumulatedDb[l][j] += db[l][j] / batch.length;
            for (let i = 0; i < nn.sizes[l]; i++) {
              accumulatedDW[l][i][j] += dW[l][i][j] / batch.length;
            }
          }
        }
      }
      
      nn.updateWeights(accumulatedDW, accumulatedDb, lr);
    }
    
    const epochLoss = totalLoss / normData.length;
    lossPoints.push(epochLoss);
    
    if (epoch % 5 === 0 || epoch === epochs) {
      updateProgress(5 + Math.round((epoch / epochs) * 45), `ML: Training Epoch ${epoch}/${epochs}`);
      
      nnLossChart.data.labels.push(epoch);
      nnLossChart.update('none');
      
      // Animate neurons
      if (nnVis && !nnVis.animating) {
        nnVis.animatePulse(nn, 'forward', 150);
      }
      
      if (epoch % 50 === 0 || epoch === epochs) {
        writeLog(`Epoch ${epoch}/${epochs} - Mean Squared Error: ${epochLoss.toFixed(6)}`, 'ml');
      }
      await sleepMs(10);
    }
  }
  
  writeLog('Neural Network training complete. Finding optimal parameters using Adam Input Gradient Ascent...', 'ml');

  // ── Step 3: Gradient Ascent on NN Inputs ──
  // Start from normalized mid-point values
  const xNorm = [0.5, 0.5, 0.5, 0.5]; 
  const momentum = [0, 0, 0, 0];
  const beta = 0.9;
  const gaLR = 0.05; // Ascent step learning rate
  
  let bestFitness = -Infinity;
  let bestXNorm = [...xNorm];
  const searchHistory = [];
  
  for (let s = 1; s <= steps; s++) {
    // 1. Forward Pass to predict gain spectrum
    const predNorm = nn.forward(xNorm);
    const predGains = predNorm.map(g => g * 40.0); // denormalize
    
    const scoreVal = calculateFitness(predGains);
    
    if (scoreVal.score > bestFitness) {
      bestFitness = scoreVal.score;
      bestXNorm = [...xNorm];
    }
    
    // 2. Compute the analytical gradient of fitness objective w.r.t gain outputs (y)
    // Objective: f = mean(y) - 3 * (max(y) - min(y)) - 2 * sqrt(mean(d2y^2))
    const N = predGains.length;
    const jMax = predGains.indexOf(Math.max(...predGains));
    const jMin = predGains.indexOf(Math.min(...predGains));
    
    const df_dy = new Array(N).fill(0);
    
    // Precompute second derivatives for smoothness gradient
    const R = new Array(N - 2).fill(0);
    let smoothnessVal = 0;
    for (let k = 1; k < N - 1; k++) {
      R[k-1] = predGains[k-1] - 2 * predGains[k] + predGains[k+1];
      smoothnessVal += R[k-1] * R[k-1];
    }
    smoothnessVal = Math.sqrt(smoothnessVal / N);
    
    for (let j = 0; j < N; j++) {
      // Mean gain derivative component
      const dMean = 1.0 / N;
      
      // Flatness derivative component
      let dFlat = 0;
      if (j === jMax) dFlat = 1.0;
      if (j === jMin) dFlat = -1.0;
      
      // Smoothness derivative component
      let dSmooth = 0;
      if (smoothnessVal > 1e-5) {
        let num = 0;
        if (j >= 2) num += R[j-2];
        if (j >= 1 && j <= N-2) num += -2 * R[j-1];
        if (j <= N-3) num += R[j];
        dSmooth = num / (N * smoothnessVal);
      }
      
      // Gradient w.r.t output gain: df/dy_j
      df_dy[j] = dMean - 3.0 * dFlat - 2.0 * dSmooth;
    }
    
    // Scale gradient back for denormalization factor (since inputs were divided by 40)
    const df_dynorm = df_dy.map(df => df * 40.0);
    
    // 3. Backpropagate error gradients from outputs to inputs (x)
    const gradX = nn.inputGradient(df_dynorm);
    
    // 4. Update normalized inputs (Gradient Ascent with Momentum)
    for (let i = 0; i < 4; i++) {
      momentum[i] = beta * momentum[i] + (1.0 - beta) * gradX[i];
      xNorm[i] += gaLR * momentum[i];
      // clamp bounds
      xNorm[i] = Math.max(0, Math.min(1, xNorm[i]));
    }
    
    // Denormalize for physical evaluation
    const realX = xNorm.map((val, idx) => {
      const [min, max] = ranges[idx];
      return min + val * (max - min);
    });
    
    const realGains = wls.map(wl => calculateNetGain(wl, realX[0], realX[1], realX[2], realX[3], selectedBand));
    const fitnessVal = calculateFitness(realGains);
    
    // Record current parameters to draw the search path trail
    searchHistory.push({
      chromosome: [...realX],
      score: fitnessVal.score
    });
    
    if (s % 5 === 0 || s === steps) {
      updateProgress(50 + Math.round((s / steps) * 50), `ML: Input Optimization ${s}/${steps}`);
      
      // Update real-time spectrum plots
      updateChartDataset(gainChart, 'ML Actual', wls.map((w, idx) => ({ x: w, y: realGains[idx] })), '#8b5cf6');
      updateChartDataset(gainChart, 'ML Predicted', wls.map((w, idx) => ({ x: w, y: predGains[idx] })), '#a78bfa', true);
      
      // Draw search trace path on the contour canvas
      if (landscapeVis) {
        const tempBest = bestXNorm.map((val, idx) => {
          const [min, max] = ranges[idx];
          return min + val * (max - min);
        });
        landscapeVis.drawParticles(searchHistory, { chromosome: tempBest }, '#8b5cf6');
      }
      
      const nf = calculateNoiseFigure(realX[0], realX[1], realX[3], selectedBand);
      refreshMetrics(fitnessVal.meanGain, fitnessVal.flatness, nf, samplesCount + s);
      
      if (nnVis && !nnVis.animating) {
        nnVis.animatePulse(nn, 'backward', 150);
      }
      await sleepMs(15);
    }
  }

  // Final Best results
  const finalBestX = bestXNorm.map((val, idx) => {
    const [min, max] = ranges[idx];
    return min + val * (max - min);
  });
  const finalBestGains = wls.map(wl => calculateNetGain(wl, finalBestX[0], finalBestX[1], finalBestX[2], finalBestX[3], selectedBand));
  const finalFitness = calculateFitness(finalBestGains);
  
  writeLog(`ML Surrogate complete! Optimal Lr=${finalBestX[0].toFixed(2)} km, Le=${finalBestX[1].toFixed(2)} m.`, 'ml');

  return {
    best: {
      chromosome: finalBestX,
      score: finalFitness.score,
      gains: finalBestGains,
      rl: finalBestX[0],
      el: finalBestX[1],
      rp: finalBestX[2],
      ep: finalBestX[3]
    },
    evals: samplesCount + steps
  };
}

/**
 * Main Trigger Orchestrator.
 */
async function runOptimization() {
  if (running) return;
  running = true;
  
  // UI States
  const btn = document.getElementById('run-btn');
  const badge = document.getElementById('status-badge');
  const indicator = document.getElementById('status-indicator');
  
  if (btn) btn.disabled = true;
  if (badge) badge.textContent = 'RUNNING';
  if (indicator) indicator.className = 'status-indicator running-ml';
  
  document.getElementById('log-box').innerHTML = '';
  document.getElementById('best-params').innerHTML = '<div style="color: var(--text-dark); text-align: center; padding: 20px;">Optimization execution in progress...</div>';
  
  const ranges = getParameterRanges();
  
  try {
    const result = await triggerML(ranges);
    
    if (result) {
      renderBestParameters(result.best, 'ML Surrogate NN & Input Gradient Ascent');
    }
    
    if (badge) badge.textContent = 'DONE';
    if (indicator) indicator.className = 'status-indicator done';
    updateProgress(100, 'Optimization complete!');
    writeLog('Amplifier parameters successfully optimized!', 'system');
    
  } catch (error) {
    console.error(error);
    writeLog(`Fatal execution error: ${error.message}`, 'system');
    if (badge) badge.textContent = 'ERROR';
    if (indicator) indicator.className = 'status-indicator';
  } finally {
    running = false;
    if (btn) btn.disabled = false;
  }
}

// ── Initial Setup ──
document.addEventListener('DOMContentLoaded', () => {
  // Setup inputs slider binds
  initializeSliders();
  
  // Setup Visualizations Canvas
  landscapeVis = new ContourLandscapeVisualizer('contourCanvas', getParameterRanges(), () => 0);
  nnVis = new NeuralNetworkVisualizer('nnCanvas');
  
  // Render empty placeholder states
  triggerLandscapeUpdate();
  initializeCharts();
  
  if (nnVis) {
    // draw initial idle NN node visualizer
    nnVis.drawNetwork(new SimpleNN(4, [16, 16], 50), 'idle');
  }
  
  writeLog('Amplifier simulator control panel initialized.', 'system');
});
