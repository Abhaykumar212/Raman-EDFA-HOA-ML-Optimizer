/**
 * nn.js
 * Machine Learning & Reinforcement Learning Algorithms
 * Contains: 
 * 1. SimpleNN - Feedforward Neural Network (Surrogate Model) from scratch with Adam
 * 2. QLearningAgent - Reinforcement Learning agent for parameter tuning
 */

class SimpleNN {
  constructor(inputSize = 4, hiddenSizes = [32, 32], outputSize = 50) {
    this.sizes = [inputSize, ...hiddenSizes, outputSize];
    this.layersCount = this.sizes.length;
    this.weights = [];
    this.biases = [];
    
    // He/Xavier Initialization
    for (let l = 0; l < this.layersCount - 1; l++) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l+1];
      const scale = Math.sqrt(2.0 / nIn);
      
      const layerW = [];
      for (let i = 0; i < nIn; i++) {
        const row = [];
        for (let j = 0; j < nOut; j++) {
          row.push((Math.random() * 2 - 1) * scale);
        }
        layerW.push(row);
      }
      this.weights.push(layerW);
      this.biases.push(new Array(nOut).fill(0));
    }
    
    // Adam optimizer status variables
    this.m_W = [];
    this.v_W = [];
    this.m_b = [];
    this.v_b = [];
    for (let l = 0; l < this.layersCount - 1; l++) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l+1];
      this.m_W.push(Array.from({length: nIn}, () => new Array(nOut).fill(0)));
      this.v_W.push(Array.from({length: nIn}, () => new Array(nOut).fill(0)));
      this.m_b.push(new Array(nOut).fill(0));
      this.v_b.push(new Array(nOut).fill(0));
    }
    this.t = 0;
  }

  // Tanh Activation for hidden layers
  activate(x) {
    return Math.tanh(x);
  }
  
  activatePrime(a) {
    return 1.0 - a * a; // derivative of tanh
  }

  /**
   * Neural network forward pass
   */
  forward(input) {
    this.activations = [input];
    this.zs = [];
    
    let current = input;
    for (let l = 0; l < this.layersCount - 1; l++) {
      const next = [];
      const z = [];
      const nIn = this.sizes[l];
      const nOut = this.sizes[l+1];
      const w = this.weights[l];
      const b = this.biases[l];
      
      for (let j = 0; j < nOut; j++) {
        let sum = b[j];
        for (let i = 0; i < nIn; i++) {
          sum += current[i] * w[i][j];
        }
        z.push(sum);
        
        if (l < this.layersCount - 2) {
          next.push(this.activate(sum)); // tanh for hidden layers
        } else {
          next.push(sum); // linear activation for output layer
        }
      }
      this.zs.push(z);
      this.activations.push(next);
      current = next;
    }
    return current;
  }

  /**
   * Neural network backward pass (backpropagation)
   */
  backward(target) {
    const outputAct = this.activations[this.layersCount - 1];
    const nOut = target.length;
    const dW = [];
    const db = [];
    
    for (let l = 0; l < this.layersCount - 1; l++) {
      dW.push(Array.from({length: this.sizes[l]}, () => new Array(this.sizes[l+1]).fill(0)));
      db.push(new Array(this.sizes[l+1]).fill(0));
    }
    
    // Output error (Mean Squared Error gradient w.r.t outputs)
    let delta = [];
    for (let j = 0; j < nOut; j++) {
      delta.push(outputAct[j] - target[j]);
    }
    
    // Backpropagate error
    for (let l = this.layersCount - 2; l >= 0; l--) {
      const act = this.activations[l];
      const nIn = this.sizes[l];
      const nNext = this.sizes[l+1];
      
      for (let j = 0; j < nNext; j++) {
        db[l][j] = delta[j];
        for (let i = 0; i < nIn; i++) {
          dW[l][i][j] = act[i] * delta[j];
        }
      }
      
      if (l > 0) {
        const nextDelta = [];
        const w = this.weights[l];
        const prevAct = this.activations[l];
        for (let i = 0; i < nIn; i++) {
          let error = 0;
          for (let j = 0; j < nNext; j++) {
            error += delta[j] * w[i][j];
          }
          nextDelta.push(error * this.activatePrime(prevAct[i]));
        }
        delta = nextDelta;
      }
    }
    return { dW, db };
  }

  /**
   * Weight updates using Adam optimizer
   */
  updateWeights(dW, db, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.t++;
    const biasCorrection1 = 1.0 - Math.pow(beta1, this.t);
    const biasCorrection2 = 1.0 - Math.pow(beta2, this.t);
    
    for (let l = 0; l < this.layersCount - 1; l++) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l+1];
      
      for (let i = 0; i < nIn; i++) {
        for (let j = 0; j < nOut; j++) {
          const g = dW[l][i][j];
          this.m_W[l][i][j] = beta1 * this.m_W[l][i][j] + (1.0 - beta1) * g;
          this.v_W[l][i][j] = beta2 * this.v_W[l][i][j] + (1.0 - beta2) * g * g;
          const mHat = this.m_W[l][i][j] / biasCorrection1;
          const vHat = this.v_W[l][i][j] / biasCorrection2;
          this.weights[l][i][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
      }
      
      for (let j = 0; j < nOut; j++) {
        const g = db[l][j];
        this.m_b[l][j] = beta1 * this.m_b[l][j] + (1.0 - beta1) * g;
        this.v_b[l][j] = beta2 * this.v_b[l][j] + (1.0 - beta2) * g * g;
        const mHat = this.m_b[l][j] / biasCorrection1;
        const vHat = this.v_b[l][j] / biasCorrection2;
        this.biases[l][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    }
  }

  /**
   * Calculates the gradient of the neural network outputs w.r.t the inputs.
   * This is used for input gradient ascent optimization.
   * 
   * @param {Array<number>} dObj_dy - Gradient of the objective function w.r.t NN outputs
   * @returns {Array<number>} Gradient of the objective function w.r.t NN inputs
   */
  inputGradient(dObj_dy) {
    let delta = [...dObj_dy];
    
    for (let l = this.layersCount - 2; l >= 0; l--) {
      const nIn = this.sizes[l];
      const nNext = this.sizes[l+1];
      const w = this.weights[l];
      const prevAct = this.activations[l];
      
      const nextDelta = [];
      for (let i = 0; i < nIn; i++) {
        let error = 0;
        for (let j = 0; j < nNext; j++) {
          error += delta[j] * w[i][j];
        }
        if (l > 0) {
          nextDelta.push(error * this.activatePrime(prevAct[i]));
        } else {
          nextDelta.push(error); // Linear input layer gradient
        }
      }
      delta = nextDelta;
    }
    return delta;
  }
}

// Export modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    SimpleNN
  };
}
