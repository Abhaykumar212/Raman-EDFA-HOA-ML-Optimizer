/**
 * optimizers.js
 * Implementation of alternative optimization algorithms:
 * 1. Genetic Algorithm (GA)
 * 2. Particle Swarm Optimization (PSO)
 * 3. Q-Learning Reinforcement Learning (RL)
 */

/**
 * 1. GENETIC ALGORITHM (GA)
 */
class GeneticAlgorithm {
  constructor(ranges, popSize = 50, crossoverRate = 0.8, mutationRate = 0.15, evaluateFn) {
    this.ranges = ranges; // [[min, max], ...]
    this.popSize = popSize;
    this.crossoverRate = crossoverRate;
    this.mutationRate = mutationRate;
    this.evaluateFn = evaluateFn; // takes [rl, el, rp, ep], returns { score, gains, ... }
    
    this.population = [];
    this.bestIndividual = null;
    this.generation = 0;
  }

  init() {
    this.population = [];
    for (let i = 0; i < this.popSize; i++) {
      const chromosome = Array.from({ length: 4 }, () => Math.random()); // normalized 0 to 1
      const realParams = this.denormalize(chromosome);
      const evalRes = this.evaluateFn(realParams);
      
      const ind = {
        chromosome: chromosome,
        score: evalRes.score,
        gains: evalRes.gains,
        fitness: Math.max(0.01, evalRes.score + 100) // offset to make fitness positive for roulette wheel
      };
      
      this.population.push(ind);
    }
    this.updateBest();
    this.generation = 0;
  }

  denormalize(chromosome) {
    return chromosome.map((val, idx) => {
      const [min, max] = this.ranges[idx];
      return min + val * (max - min);
    });
  }

  updateBest() {
    let best = this.population[0];
    for (let i = 1; i < this.popSize; i++) {
      if (this.population[i].score > best.score) {
        best = this.population[i];
      }
    }
    // Deep copy
    this.bestIndividual = {
      chromosome: [...best.chromosome],
      score: best.score,
      gains: [...best.gains],
      realParams: this.denormalize(best.chromosome)
    };
  }

  // Tournament selection
  selectParent(tournamentSize = 3) {
    let best = this.population[Math.floor(Math.random() * this.popSize)];
    for (let i = 1; i < tournamentSize; i++) {
      const candidate = this.population[Math.floor(Math.random() * this.popSize)];
      if (candidate.score > best.score) {
        best = candidate;
      }
    }
    return best;
  }

  // Blend crossover (BLX-alpha)
  crossover(parentA, parentB) {
    if (Math.random() > this.crossoverRate) {
      return [...parentA.chromosome];
    }
    
    const childChrom = [];
    const alpha = 0.2;
    for (let j = 0; j < 4; j++) {
      const valA = parentA.chromosome[j];
      const valB = parentB.chromosome[j];
      
      const min = Math.min(valA, valB);
      const max = Math.max(valA, valB);
      const diff = max - min;
      
      // Select random value in range [min - diff*alpha, max + diff*alpha]
      const rangeMin = Math.max(0, min - diff * alpha);
      const rangeMax = Math.min(1, max + diff * alpha);
      childChrom.push(rangeMin + Math.random() * (rangeMax - rangeMin));
    }
    return childChrom;
  }

  // Polynomial/Gaussian mutation
  mutate(chromosome) {
    return chromosome.map(val => {
      if (Math.random() > this.mutationRate) return val;
      // Perturb by small gaussian-like step
      const perturbation = (Math.random() + Math.random() + Math.random() - 1.5) * 0.15;
      return Math.max(0, Math.min(1, val + perturbation));
    });
  }

  step() {
    this.generation++;
    const nextPop = [];
    
    // Elitism: carry over top 2 individuals directly
    const sorted = [...this.population].sort((a, b) => b.score - a.score);
    nextPop.push(sorted[0]);
    nextPop.push(sorted[1]);
    
    while (nextPop.length < this.popSize) {
      const p1 = this.selectParent();
      const p2 = this.selectParent();
      
      let childChrom = this.crossover(p1, p2);
      childChrom = this.mutate(childChrom);
      
      const realParams = this.denormalize(childChrom);
      const evalRes = this.evaluateFn(realParams);
      
      nextPop.push({
        chromosome: childChrom,
        score: evalRes.score,
        gains: evalRes.gains,
        fitness: Math.max(0.01, evalRes.score + 100)
      });
    }
    
    this.population = nextPop;
    this.updateBest();
    
    return {
      generation: this.generation,
      best: this.bestIndividual,
      population: this.population.map(ind => ({
        chromosome: this.denormalize(ind.chromosome),
        score: ind.score
      }))
    };
  }
}

/**
 * 2. PARTICLE SWARM OPTIMIZATION (PSO)
 */
class ParticleSwarmOptimization {
  constructor(ranges, swarmSize = 40, iterations = 80, c1 = 1.8, c2 = 1.8, evaluateFn) {
    this.ranges = ranges;
    this.swarmSize = swarmSize;
    this.iterations = iterations;
    this.c1 = c1; // cognitive factor
    this.c2 = c2; // social factor
    this.evaluateFn = evaluateFn;
    
    this.particles = [];
    this.gBestPosition = null;
    this.gBestScore = -Infinity;
    this.gBestGains = [];
    this.iteration = 0;
  }

  init() {
    this.particles = [];
    this.gBestScore = -Infinity;
    this.gBestPosition = null;
    this.gBestGains = [];
    this.iteration = 0;

    for (let i = 0; i < this.swarmSize; i++) {
      const pos = Array.from({ length: 4 }, () => Math.random()); // normalized
      const vel = Array.from({ length: 4 }, () => (Math.random() * 2 - 1) * 0.1); // speed
      
      const realParams = this.denormalize(pos);
      const evalRes = this.evaluateFn(realParams);
      
      const particle = {
        position: pos,
        velocity: vel,
        score: evalRes.score,
        gains: evalRes.gains,
        pBestPosition: [...pos],
        pBestScore: evalRes.score
      };
      
      this.particles.push(particle);
      
      if (evalRes.score > this.gBestScore) {
        this.gBestScore = evalRes.score;
        this.gBestPosition = [...pos];
        this.gBestGains = [...evalRes.gains];
      }
    }
  }

  denormalize(pos) {
    return pos.map((val, idx) => {
      const [min, max] = this.ranges[idx];
      return min + val * (max - min);
    });
  }

  step() {
    this.iteration++;
    
    // Inertia weight drops over time (high exploration initially, high exploitation later)
    const w = 0.9 - 0.5 * (this.iteration / this.iterations);
    
    for (let i = 0; i < this.swarmSize; i++) {
      const p = this.particles[i];
      
      for (let j = 0; j < 4; j++) {
        const r1 = Math.random();
        const r2 = Math.random();
        
        // PSO Velocity Update Equation
        p.velocity[j] = w * p.velocity[j] 
                        + this.c1 * r1 * (p.pBestPosition[j] - p.position[j])
                        + this.c2 * r2 * (this.gBestPosition[j] - p.position[j]);
        
        // Max velocity clamp
        p.velocity[j] = Math.max(-0.2, Math.min(0.2, p.velocity[j]));
        
        // Position update
        p.position[j] = Math.max(0, Math.min(1, p.position[j] + p.velocity[j]));
      }
      
      // Evaluate particle
      const realParams = this.denormalize(p.position);
      const evalRes = this.evaluateFn(realParams);
      p.score = evalRes.score;
      p.gains = evalRes.gains;
      
      // Update pBest
      if (p.score > p.pBestScore) {
        p.pBestScore = p.score;
        p.pBestPosition = [...p.position];
      }
      
      // Update gBest
      if (p.score > this.gBestScore) {
        this.gBestScore = p.score;
        this.gBestPosition = [...p.position];
        this.gBestGains = [...p.gains];
      }
    }
    
    return {
      iteration: this.iteration,
      best: {
        chromosome: this.gBestPosition,
        score: this.gBestScore,
        gains: this.gBestGains,
        realParams: this.denormalize(this.gBestPosition)
      },
      particles: this.particles.map(p => ({
        chromosome: this.denormalize(p.position),
        score: p.score
      }))
    };
  }
}

/**
 * 3. Q-LEARNING REINFORCEMENT LEARNING AGENT
 */
class QLearningOptimizer {
  constructor(ranges, episodes = 200, stepsPerEpisode = 15, alpha = 0.15, gamma = 0.85, epsilon = 0.25, evaluateFn) {
    this.ranges = ranges;
    this.episodes = episodes;
    this.stepsPerEpisode = stepsPerEpisode;
    this.alpha = alpha; // learning rate
    this.gamma = gamma; // discount factor
    this.epsilon = epsilon; // exploration rate
    this.evaluateFn = evaluateFn;
    
    this.bins = 6; // discretize each dimensions into 6 bins (0, 0.2, 0.4, 0.6, 0.8, 1.0)
    this.qTable = {}; // State -> array of 9 actions values
    
    // Actions:
    // 0: stay, 1/2: rl up/down, 3/4: el up/down, 5/6: rp up/down, 7/8: ep up/down
    this.actions = [
      [0, 0, 0, 0],
      [0.1, 0, 0, 0], [-0.1, 0, 0, 0],
      [0, 0.1, 0, 0], [0, -0.1, 0, 0],
      [0, 0, 0.1, 0], [0, 0, -0.1, 0],
      [0, 0, 0, 0.1], [0, 0, 0, -0.1]
    ];
    
    this.currentState = [0.5, 0.5, 0.5, 0.5]; // starts at midpoint
    this.bestScore = -Infinity;
    this.bestPosition = null;
    this.bestGains = [];
    this.episode = 0;
  }

  init() {
    this.qTable = {};
    this.bestScore = -Infinity;
    this.bestPosition = null;
    this.bestGains = [];
    this.episode = 0;
    this.currentState = [0.5, 0.5, 0.5, 0.5];
  }

  // Get state string based on discretized coordinate index
  getStateKey(pos) {
    const indices = pos.map(val => Math.max(0, Math.min(this.bins - 1, Math.round(val * (this.bins - 1)))));
    return indices.join(',');
  }

  getQValues(stateKey) {
    if (!this.qTable[stateKey]) {
      this.qTable[stateKey] = new Array(this.actions.length).fill(0.0);
    }
    return this.qTable[stateKey];
  }

  denormalize(pos) {
    return pos.map((val, idx) => {
      const [min, max] = this.ranges[idx];
      return min + val * (max - min);
    });
  }

  step() {
    this.episode++;
    
    // Decay exploration over time
    const eps = this.epsilon * (1.0 - this.episode / this.episodes);
    
    // Start at a random spot or current best to promote local exploration
    let currentPos = this.bestPosition ? [...this.bestPosition].map(v => Math.max(0, Math.min(1, v + (Math.random() * 2 - 1) * 0.1))) : [Math.random(), Math.random(), Math.random(), Math.random()];
    let currentEval = this.evaluateFn(this.denormalize(currentPos));
    let currentScore = currentEval.score;

    if (currentScore > this.bestScore) {
      this.bestScore = currentScore;
      this.bestPosition = [...currentPos];
      this.bestGains = [...currentEval.gains];
    }
    
    const trail = [];

    for (let step = 0; step < this.stepsPerEpisode; step++) {
      const stateKey = this.getStateKey(currentPos);
      const qValues = this.getQValues(stateKey);
      
      // Select Action (Epsilon-Greedy)
      let actionIdx = 0;
      if (Math.random() < eps) {
        actionIdx = Math.floor(Math.random() * this.actions.length);
      } else {
        // Find action with max Q value
        let maxQ = -Infinity;
        for (let a = 0; a < qValues.length; a++) {
          if (qValues[a] > maxQ) {
            maxQ = qValues[a];
            actionIdx = a;
          }
        }
      }
      
      const act = this.actions[actionIdx];
      
      // Calculate new position
      const nextPos = currentPos.map((val, idx) => Math.max(0, Math.min(1, val + act[idx])));
      const nextEval = this.evaluateFn(this.denormalize(nextPos));
      const nextScore = nextEval.score;
      
      // Reward function: directly based on fitness improvement
      const reward = nextScore + 100; // offset to guarantee positive reinforcement
      
      // Bellman Q-Value Update
      const nextStateKey = this.getStateKey(nextPos);
      const nextQValues = this.getQValues(nextStateKey);
      const maxNextQ = Math.max(...nextQValues);
      
      qValues[actionIdx] = qValues[actionIdx] + this.alpha * (reward + this.gamma * maxNextQ - qValues[actionIdx]);
      
      // Move agent
      currentPos = nextPos;
      currentScore = nextScore;
      
      if (currentScore > this.bestScore) {
        this.bestScore = currentScore;
        this.bestPosition = [...currentPos];
        this.bestGains = [...nextEval.gains];
      }
      
      trail.push({
        chromosome: this.denormalize(currentPos),
        score: currentScore
      });
    }

    return {
      episode: this.episode,
      best: {
        chromosome: this.bestPosition,
        score: this.bestScore,
        gains: this.bestGains,
        realParams: this.denormalize(this.bestPosition)
      },
      trail: trail // Returns the steps of the agent in this episode to animate
    };
  }
}

// Export classes for module checks
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    QLearningOptimizer
  };
}
