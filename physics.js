/**
 * physics.js
 * Raman-EDFA Hybrid Optical Amplifier Simulation Physics Engine
 * Based on Singh & Kaler (2014) Rate Equations
 */

// Global Constants
const H_PLANCK = 6.626e-34; // J*s
const C_LIGHT = 2.998e8;    // m/s

// Bands configurations
const BANDS = {
  'c-band': {
    name: 'C-band (Standard C-Band Grid)',
    startWl: 1530.33,
    endWl: 1565.0,
    defaultSpacing: 0.8, // 100 GHz spacing
    defaultChannels: 44,
    pumpRamanWl: 1453.0, // peak gain around 1550 nm
    pumpEdfaWl: 980.0
  },
  'l-band': {
    name: 'L-band (Singh & Kaler 2014 Grid)',
    startWl: 1624.0,
    endWl: 1634.0,
    defaultSpacing: 0.2, // 0.2 nm spacing
    defaultChannels: 50,
    pumpRamanWl: 1530.0, // peak gain around 1630 nm
    pumpEdfaWl: 980.0
  }
};

/**
 * Calculates the signal frequency (Hz) from wavelength (nm).
 */
function wlToFreq(wl) {
  return C_LIGHT / (wl * 1e-9);
}

/**
 * Raman Amplifier Model
 * Derived from paper equations (7) - (11):
 * Forward pumping scheme, pump depletion neglected.
 * 
 * @param {number} wl - Signal wavelength (nm)
 * @param {number} rl - Raman fiber length (km)
 * @param {number} rp - Raman pump power (mW)
 * @param {string} bandId - 'c-band' or 'l-band'
 */
function calculateRamanGain(wl, rl, rp, bandId) {
  const band = BANDS[bandId] || BANDS['c-band'];
  const rl_m = rl * 1000;      // km -> m
  const rp_w = rp / 1000;      // mW -> W

  // Fiber physical parameters
  const aP = 55e-12;           // Pump effective area (m^2)
  const alpha_p_db = 0.25;     // Pump fiber loss (dB/km)
  const alpha_s_db = 0.20;     // Signal fiber loss (dB/km)
  
  // Convert dB/km to linear loss (m^-1)
  const alpha_p = (alpha_p_db / 4.343) / 1000;
  const alpha_s = (alpha_s_db / 4.343) / 1000;

  // Raman gain coefficient shape gR(lambda) in m/W
  // Peak gain is about 1e-13 m/W at ~100 nm shift from pump
  const pumpWl = band.pumpRamanWl;
  const shift_nm = wl - pumpWl;
  
  // Model Raman gain profile of standard silica fiber (peaks at ~100nm frequency shift)
  const peakShift = 97.0; 
  const width = 35.0;     // width of gain curve
  const gR_peak = 1.1e-13; // peak coefficient (m/W)
  
  const gR = gR_peak * Math.exp(-Math.pow(shift_nm - peakShift, 2) / (2 * Math.pow(width, 2)));

  // Eq.(11) simplified:
  // C = (gR * PPin-R * (1 - exp(-alpha_p * L))) / (aP * alpha_p * L) - alpha_s
  // Net gain exponent is C * L
  let exponent = 0;
  if (rl_m > 0) {
    const term1 = (gR * rp_w) / (aP * alpha_p);
    const term2 = 1.0 - Math.exp(-alpha_p * rl_m);
    exponent = (term1 * term2) - (alpha_s * rl_m);
  } else {
    exponent = 0;
  }

  // Net Raman gain factor (linear)
  const gainLinear = Math.exp(exponent);
  
  // Convert to dB
  const gainDb = 10 * Math.log10(gainLinear);
  return { gainDb, gainLinear };
}

/**
 * EDFA Amplifier Model
 * Derived from paper equations (1) - (6):
 * Steady-state Erbium population distribution in a two-level system.
 * 
 * @param {number} wl - Signal wavelength (nm)
 * @param {number} el - EDFA fiber length (m)
 * @param {number} ep - EDFA pump power (mW)
 * @param {number} inputSignalPowerLinear - Input signal power from Raman stage (W)
 * @param {string} bandId - 'c-band' or 'l-band'
 */
function calculateEdfaGain(wl, el, ep, inputSignalPowerLinear, bandId) {
  const band = BANDS[bandId] || BANDS['c-band'];
  const ep_w = ep / 1000; // mW -> W

  // Fiber physical parameters
  const Nt = 1.2e25;      // Erbium ion density (m^-3)
  const ap = 20e-12;      // Pump effective mode area (m^2)
  const as = 20e-12;      // Signal effective mode area (m^2)
  const tau = 10e-3;      // Spontaneous emission lifetime (10 ms)
  
  // Absorption/emission cross-sections (m^2)
  const sigma_pa = 2.5e-25; // Pump absorption cross-section at 980 nm
  
  // Signal cross-sections depending on band and wavelength
  let sigma_sa = 0;
  let sigma_se = 0;

  if (bandId === 'c-band') {
    // Normal C-band shape peaking near 1530 nm and 1550 nm
    const wlNorm1 = (wl - 1530) / 10;
    const wlNorm2 = (wl - 1550) / 15;
    
    sigma_sa = 6.2e-25 * Math.exp(-Math.pow(wlNorm1, 2)) + 2.5e-25 * Math.exp(-Math.pow(wlNorm2, 2));
    sigma_se = 6.0e-25 * Math.exp(-Math.pow(wlNorm1, 2)) + 3.4e-25 * Math.exp(-Math.pow(wlNorm2, 2));
  } else {
    // L-band shape (1620-1640 nm). Erbium transition is much weaker here
    // Singh & Kaler (2014) L-band profile
    const wlNorm = (wl - 1600) / 30;
    sigma_sa = 0.5e-25 * Math.exp(-Math.pow(wlNorm - 0.5, 2) / 0.5);
    sigma_se = 0.8e-25 * Math.exp(-Math.pow(wlNorm - 0.5, 2) / 0.4);
  }

  // Calculate Transition Rates
  const nu_p = wlToFreq(band.pumpEdfaWl);
  const nu_s = wlToFreq(wl);

  const Rp = (sigma_pa * ep_w) / (H_PLANCK * nu_p * ap);
  const Rs = (sigma_sa * inputSignalPowerLinear) / (H_PLANCK * nu_s * as);
  const Rse = (sigma_se * inputSignalPowerLinear) / (H_PLANCK * nu_s * as);

  // Steady-state population fractions (n1 = N1/Nt, n2 = N2/Nt)
  // rate eqn: dN2/dt = Rp*N1 + Rs*N1 - Rse*N2 - N2/tau = 0
  const denom = Rp + Rs + Rse + (1.0 / tau);
  const n2 = (Rp + Rs) / denom;
  const n1 = 1.0 - n2;

  // Signal propagation along EDFA length: dPs/dz = (sigma_se * N2 - sigma_sa * N1) * Ps
  const gainCoeff = (sigma_se * n2 - sigma_sa * n1) * Nt;
  const gainLinear = Math.exp(gainCoeff * el);
  const gainDb = 10 * Math.log10(gainLinear);

  return { gainDb, gainLinear };
}

/**
 * Calculates Net HOA Gain
 * Eq.(12): Net gain accounts for coupling losses and Cascaded Amplification.
 */
function calculateNetGain(wl, rl, el, rp, ep, bandId, perChannelInputDbm = -10) {
  const pin_w = Math.pow(10, (perChannelInputDbm - 30) / 10); // dBm -> W

  // Step 1: Raman Stage
  const raman = calculateRamanGain(wl, rl, rp, bandId);
  const pOutRaman = pin_w * raman.gainLinear;

  // Step 2: Coupler insertion loss
  const couplerLossDb = 0.5; // 0.5 dB loss
  const pInEdfa = pOutRaman * Math.pow(10, -couplerLossDb / 10);

  // Step 3: EDFA Stage
  const edfa = calculateEdfaGain(wl, el, ep, pInEdfa, bandId);

  // Net Gain (dB)
  const netGainDb = raman.gainDb - couplerLossDb + edfa.gainDb;
  return netGainDb;
}

/**
 * Calculates Noise Figure (NF) of the HOA.
 * Includes EDFA ASE noise + Raman ASE noise penalty.
 */
function calculateNoiseFigure(rl, el, ep, bandId) {
  // EDFA noise figure: NF = 10*log10(2 * nsp)
  // Spontaneous emission factor nsp is lower for higher pump powers
  const nspEDFA = 1.4 + 0.6 * Math.exp(-ep / 150);
  const nfEDFA = 10 * Math.log10(2 * nspEDFA);

  // Raman noise figure penalty (related to Raman fiber length & attenuation)
  const ramanNF = 0.25 * (rl / 100); 

  const netNF = nfEDFA + ramanNF;
  return +netNF.toFixed(2);
}

/**
 * Generates the Wavelength array based on band ID, number of channels, and spacing.
 */
function generateWavelengthGrid(bandId, channelsCount, spacing) {
  const band = BANDS[bandId] || BANDS['c-band'];
  const wls = [];
  for (let i = 0; i < channelsCount; i++) {
    wls.push(+(band.startWl + i * spacing).toFixed(2));
  }
  return wls;
}

/**
 * Fitness Function to maximize
 * Maximize overall gain, minimize gain flatness variation, and keep gain curve smooth.
 * 
 * score = meanGain - W_flat * Flatness - W_smooth * Smoothness
 */
function calculateFitness(gains, wFlat = 3.0, wSmooth = 2.0) {
  const meanGain = gains.reduce((sum, g) => sum + g, 0) / gains.length;
  
  const minG = Math.min(...gains);
  const maxG = Math.max(...gains);
  const flatness = maxG - minG;

  // Smoothness: second derivative roughness sum
  let smoothness = 0;
  for (let i = 1; i < gains.length - 1; i++) {
    const diff2 = gains[i-1] - 2 * gains[i] + gains[i+1];
    smoothness += diff2 * diff2;
  }
  smoothness = Math.sqrt(smoothness / gains.length);

  // Joint Objective Function
  const score = meanGain - (wFlat * flatness) - (wSmooth * smoothness);

  return {
    score,
    meanGain,
    flatness,
    smoothness
  };
}

// Export for usage in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    BANDS,
    wlToFreq,
    calculateRamanGain,
    calculateEdfaGain,
    calculateNetGain,
    calculateNoiseFigure,
    generateWavelengthGrid,
    calculateFitness
  };
}
