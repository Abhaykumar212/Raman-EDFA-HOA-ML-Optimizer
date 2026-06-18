/**
 * visualizer.js
 * Canvas Visualizations for:
 * 1. 2D Contour Landscape Plot (GA/PSO search space)
 * 2. Neural Network Node Visualizer (Training forward/backprop animation)
 */

class ContourLandscapeVisualizer {
  constructor(canvasId, ranges, evaluateFn) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.ranges = ranges; // [[min, max], ...]
    this.evaluateFn = evaluateFn;
    
    // Default axes: X = Raman Length (idx 0), Y = EDFA Length (idx 1)
    this.xAxisIdx = 0;
    this.yAxisIdx = 1;
    
    this.cacheGrid = null;
    this.cacheOtherParams = [];
  }

  /**
   * Generates a cache of the fitness grid to make redraws of particles fast.
   */
  generateLandscapeCache(allCurrentParams, resolution = 25) {
    this.cacheOtherParams = [...allCurrentParams];
    const grid = [];
    
    const xRange = this.ranges[this.xAxisIdx];
    const yRange = this.ranges[this.yAxisIdx];
    
    for (let y = 0; y < resolution; y++) {
      const row = [];
      const yVal = yRange[0] + (y / (resolution - 1)) * (yRange[1] - yRange[0]);
      for (let x = 0; x < resolution; x++) {
        const xVal = xRange[0] + (x / (resolution - 1)) * (xRange[1] - xRange[0]);
        
        // Build parameter array
        const params = [...allCurrentParams];
        params[this.xAxisIdx] = xVal;
        params[this.yAxisIdx] = yVal;
        
        const res = this.evaluateFn(params);
        row.push(res.score);
      }
      grid.push(row);
    }
    
    this.cacheGrid = grid;
  }

  /**
   * Renders the background heat map using the cache.
   */
  drawBackground() {
    if (!this.cacheGrid) return;
    
    const w = this.canvas.width;
    const h = this.canvas.height;
    
    const resY = this.cacheGrid.length;
    const resX = this.cacheGrid[0].length;
    
    const cellW = w / resX;
    const cellH = h / resY;
    
    // Find min and max scores in cache to normalize colors
    let minScore = Infinity;
    let maxScore = -Infinity;
    for (let r = 0; r < resY; r++) {
      for (let c = 0; c < resX; c++) {
        const score = this.cacheGrid[r][c];
        if (score < minScore) minScore = score;
        if (score > maxScore) maxScore = score;
      }
    }
    
    const scoreRange = maxScore - minScore + 1e-9;
    
    // Draw grid pixels
    for (let r = 0; r < resY; r++) {
      // Draw from bottom to top on canvas because Y axis goes upwards in physics
      const yPos = h - (r + 1) * cellH; 
      for (let c = 0; c < resX; c++) {
        const xPos = c * cellW;
        
        const score = this.cacheGrid[r][c];
        const normalized = (score - minScore) / scoreRange; // 0.0 to 1.0
        
        // Custom HSL Gradient: Deep Blue (low) -> Purple -> Crimson -> Gold (high)
        // Hue goes from 240 (blue) to 360/0 (red) to 45 (gold)
        let hue = 240 - normalized * 240; 
        if (hue < 0) hue += 360;
        
        // Saturated colors for dark mode wow factor
        this.ctx.fillStyle = `hsl(${hue}, 85%, ${20 + normalized * 30}%)`; 
        this.ctx.fillRect(xPos - 0.5, yPos - 0.5, cellW + 1, cellH + 1);
      }
    }
    
    // Add grid line overlay
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    this.ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const x = (w / 4) * i;
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, h);
      this.ctx.stroke();
      
      const y = (h / 4) * i;
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(w, y);
      this.ctx.stroke();
    }
  }

  /**
   * Draws optimization swarm/population on top of the heat map.
   * 
   * @param {Array} population - Array of individuals: { chromosome: [rl, el, rp, ep], score: number }
   * @param {Object} best - Best individual { chromosome: [rl, el, rp, ep] }
   * @param {string} particleColor - Color hex string
   */
  drawParticles(population, best, particleColor = '#10B981') {
    if (!this.canvas) return;
    
    const w = this.canvas.width;
    const h = this.canvas.height;
    
    // Clear & Redraw BG
    this.ctx.clearRect(0, 0, w, h);
    this.drawBackground();
    
    const xRange = this.ranges[this.xAxisIdx];
    const yRange = this.ranges[this.yAxisIdx];
    
    const toCanvasX = (val) => ((val - xRange[0]) / (xRange[1] - xRange[0])) * w;
    // Invert Y axis so higher values are at the top
    const toCanvasY = (val) => h - (((val - yRange[0]) / (yRange[1] - yRange[0])) * h); 
    
    // Draw population dots
    this.ctx.shadowBlur = 4;
    this.ctx.shadowColor = particleColor;
    
    population.forEach(ind => {
      const px = ind.chromosome[this.xAxisIdx];
      const py = ind.chromosome[this.yAxisIdx];
      
      const cx = toCanvasX(px);
      const cy = toCanvasY(py);
      
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
      this.ctx.fillStyle = particleColor;
      this.ctx.fill();
    });
    
    // Draw global best marker
    if (best) {
      const bx = best.chromosome[this.xAxisIdx];
      const by = best.chromosome[this.yAxisIdx];
      
      const cx = toCanvasX(bx);
      const cy = toCanvasY(by);
      
      // Outer pulse ring
      this.ctx.shadowBlur = 12;
      this.ctx.shadowColor = '#F59E0B';
      this.ctx.strokeStyle = '#F59E0B';
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
      this.ctx.stroke();
      
      // Inner solid point
      this.ctx.fillStyle = '#FFFFFF';
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      this.ctx.fill();
    }
    
    // Reset shadow
    this.ctx.shadowBlur = 0;
  }
}

class NeuralNetworkVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    
    this.pulseProgress = 0;
    this.pulseDir = 1; // 1: forward, -1: backward, 0: idle
    this.animating = false;
  }

  /**
   * Renders the MLP layers graph.
   * 
   * @param {SimpleNN} nn - SimpleNN instance containing weights/biases
   * @param {string} pulseMode - 'forward', 'backward', or 'idle'
   */
  drawNetwork(nn, pulseMode = 'idle') {
    if (!this.canvas || !nn) return;
    
    const w = this.canvas.width;
    const h = this.canvas.height;
    
    this.ctx.clearRect(0, 0, w, h);
    
    // Setup node layout dimensions
    // To fit nice, we cap drawn hidden nodes
    const maxDrawnHidden = 6;
    const sizesToDraw = nn.sizes.map((s, idx) => {
      if (idx > 0 && idx < nn.sizes.length - 1) {
        return Math.min(s, maxDrawnHidden);
      }
      // Output is large (e.g. 50), draw 8 representatively
      if (idx === nn.sizes.length - 1) {
        return 8; 
      }
      return s;
    });
    
    const layersCount = sizesToDraw.length;
    const layerSpacing = w / (layersCount + 0.6);
    
    const nodeCoords = [];
    
    // Step 1: Calculate coordinates for all node representations
    for (let l = 0; l < layersCount; l++) {
      const count = sizesToDraw[l];
      const x = layerSpacing * (l + 0.8);
      const layerCoords = [];
      
      const vSpacing = h / (count + 1);
      for (let n = 0; n < count; n++) {
        const y = vSpacing * (n + 1);
        layerCoords.push({ x, y });
      }
      nodeCoords.push(layerCoords);
    }
    
    // Step 2: Draw Synaptic Connections (Weights)
    for (let l = 0; l < layersCount - 1; l++) {
      const currentLayer = nodeCoords[l];
      const nextLayer = nodeCoords[l+1];
      const actualWeights = nn.weights[l];
      
      for (let i = 0; i < currentLayer.length; i++) {
        const nodeA = currentLayer[i];
        for (let j = 0; j < nextLayer.length; j++) {
          const nodeB = nextLayer[j];
          
          // Get representative weight
          let wVal = 0;
          if (actualWeights && actualWeights[i]) {
            wVal = actualWeights[i][j] || 0;
          }
          
          const absW = Math.abs(wVal);
          const opacity = Math.min(0.35, absW * 0.4);
          
          // Color coding: Blue for positive weights, Purple/Red for negative weights
          this.ctx.strokeStyle = wVal > 0 ? `rgba(55, 138, 221, ${opacity})` : `rgba(239, 159, 39, ${opacity})`;
          this.ctx.lineWidth = Math.min(3, 0.5 + absW * 1.5);
          
          this.ctx.beginPath();
          this.ctx.moveTo(nodeA.x, nodeA.y);
          this.ctx.lineTo(nodeB.x, nodeB.y);
          this.ctx.stroke();
        }
      }
    }
    
    // Step 3: Draw Pulse Flow Animations (Adam gradient descent / Backprop)
    if (pulseMode !== 'idle') {
      this.ctx.lineWidth = 3;
      
      for (let l = 0; l < layersCount - 1; l++) {
        const currentLayer = nodeCoords[l];
        const nextLayer = nodeCoords[l+1];
        
        currentLayer.forEach((nodeA) => {
          nextLayer.forEach((nodeB) => {
            // Calculate pulse position
            let px = 0;
            let py = 0;
            
            if (pulseMode === 'forward') {
              px = nodeA.x + (nodeB.x - nodeA.x) * this.pulseProgress;
              py = nodeA.y + (nodeB.y - nodeA.y) * this.pulseProgress;
              this.ctx.fillStyle = 'rgba(16, 185, 129, 0.7)'; // Glowing Green flow
            } else if (pulseMode === 'backward') {
              px = nodeB.x + (nodeA.x - nodeB.x) * this.pulseProgress;
              py = nodeB.y + (nodeA.y - nodeB.y) * this.pulseProgress;
              this.ctx.fillStyle = 'rgba(139, 92, 246, 0.7)'; // Glowing Purple backprop
            }
            
            this.ctx.beginPath();
            this.ctx.arc(px, py, 1.5, 0, 2 * Math.PI);
            this.ctx.fill();
          });
        });
      }
    }
    
    // Step 4: Draw Neurons (Nodes)
    for (let l = 0; l < layersCount; l++) {
      const layer = nodeCoords[l];
      const originalCount = nn.sizes[l];
      const count = sizesToDraw[l];
      
      layer.forEach((node, n) => {
        // Neon borders
        this.ctx.shadowBlur = 6;
        if (l === 0) {
          this.ctx.fillStyle = '#378ADD'; // Input Blue
          this.ctx.shadowColor = '#378ADD';
        } else if (l === layersCount - 1) {
          this.ctx.fillStyle = '#9B51E0'; // Output Violet
          this.ctx.shadowColor = '#9B51E0';
        } else {
          this.ctx.fillStyle = '#4B5563'; // Hidden Grey
          this.ctx.shadowColor = '#4B5563';
        }
        
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, 6, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Inner white dot
        this.ctx.shadowBlur = 0;
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, 2, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Text labels for inputs & outputs
        if (l === 0) {
          const labels = ['RL', 'EL', 'RP', 'EP'];
          this.ctx.font = '8px monospace';
          this.ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
          this.ctx.fillText(labels[n] || '', node.x - 22, node.y + 3);
        } else if (l === layersCount - 1 && n === 0) {
          this.ctx.font = '7px monospace';
          this.ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
          this.ctx.fillText('Gain Spectrum (50 Channels)', node.x + 10, node.y - 4);
        }
      });
      
      // Draw vertical ellipsis if hidden nodes were capped
      if (originalCount > maxDrawnHidden && l > 0 && l < layersCount - 1) {
        const midY = h / 2;
        this.ctx.fillStyle = 'rgba(255,255,255,0.4)';
        this.ctx.font = '10px sans-serif';
        this.ctx.fillText('...', layer[0].x - 3, midY);
      }
    }
  }

  /**
   * Starts a smooth animation loop for active training forward/backward passes.
   */
  animatePulse(nn, mode = 'forward', duration = 300) {
    if (this.animating) return;
    this.animating = true;
    this.pulseProgress = 0;
    
    const startTime = performance.now();
    
    const loop = (now) => {
      const elapsed = now - startTime;
      this.pulseProgress = Math.min(1.0, elapsed / duration);
      
      this.drawNetwork(nn, mode);
      
      if (this.pulseProgress < 1.0) {
        requestAnimationFrame(loop);
      } else {
        this.animating = false;
        this.drawNetwork(nn, 'idle'); // clean reset
      }
    };
    
    requestAnimationFrame(loop);
  }
}

// Export classes
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    ContourLandscapeVisualizer,
    NeuralNetworkVisualizer
  };
}
