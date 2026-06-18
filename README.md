# Raman-EDFA HOA ML Optimizer

A web-based simulator and machine learning optimizer for a **Raman-EDFA Hybrid Optical Amplifier (HOA)**. This project is based on the research paper: *Singh & Kaler (2014), Journal of the Optical Society of Korea*.

---

## What is this Project About? (In Simple Words)

When internet data travels through long glass cables (optical fibers) as light waves, the signal gets weaker over distance. To keep the signal strong, we use **Optical Amplifiers** to boost it.

This project combines two different types of amplifiers to make a **Hybrid Optical Amplifier (HOA)**:
1. **Raman Amplifier:** Boosts the signal inside the transmission fiber itself.
2. **EDFA (Erbium-Doped Fiber Amplifier):** Boosts the signal using a special active fiber.

By combining them, we get a stronger and flatter boost (gain) across different light wavelengths (colors).

### The Challenge:
Finding the perfect settings (fiber lengths and pump powers) to get a **high and flat gain** is very difficult and slow using standard math equations.

### The Solution in this Project:
We use **Machine Learning (ML)**! 
1. We train a **Neural Network** to learn the complex physics equations.
2. Once trained, the Neural Network can predict how the amplifier will perform in milliseconds.
3. We then use **Gradient Ascent (an optimization algorithm)** to automatically find the best settings for the amplifier.

---

## Key Features

1. **Physical Simulation:** Runs real-time steady-state calculations for Raman and EDFA physics.
2. **2D Fitness Landscape:** A colorful 2D map showing how the amplifier's score (fitness) changes with different Raman and EDFA lengths.
3. **Neural Network Visualizer:** A live visual drawing of the neural network structure showing:
   - **4 Inputs:** Raman Length, EDFA Length, Raman Pump Power, EDFA Pump Power.
   - **2 Hidden Layers:** Neurons and their weight connections.
   - **Outputs:** The predicted gain values for all wavelength channels.
   - **Live Pulse Animations:** Green pulses showing feedforward training, and purple pulses showing backpropagation/optimization.
4. **Gain Spectrum Chart:** Displays the target gain (20 dB) alongside the predicted and actual gain across all wavelengths.
5. **Training Loss Plot:** Shows the error (Loss) of the neural network dropping in real-time as it learns.

---

## How the Machine Learning Optimization Works

The optimization process runs in three simple steps when you click **"Trigger ML Optimization"**:

1. **Step A: Generate Data**
   - The program randomly tries hundreds of configurations and calculates the output using physics equations to create a dataset.

2. **Step B: Train the Neural Network**
   - The Neural Network (4 inputs $\rightarrow$ 16 hidden nodes $\rightarrow$ 16 hidden nodes $\rightarrow$ 50 outputs) trains on this dataset using the **Adam Optimizer** to minimize error (Loss). You will see the network pulse and learn.

3. **Step C: Gradient Ascent Search**
   - The algorithm starts at a medium setting and calculates the gradient (slope) of the gain flatness. It adjusts the inputs step-by-step to reach the highest point on the fitness landscape (the best configuration).

---

## How to Run the Project Locally

You do not need to install complex databases or frameworks. You can run this project in two simple ways:

### Method 1: Just Double-Click (Easiest)
1. Download or clone this folder.
2. Double-click the `index.html` file. It will open in your web browser immediately.

### Method 2: Use a Local Server (Recommended)
Running a local server is better for loading files and scripts cleanly.

If you have **Python** installed:
1. Open your terminal/command prompt in the project folder.
2. Run this command:
   ```bash
   python -m http.server 8000
   ```
3. Open your browser and go to: **[http://localhost:8000](http://localhost:8000)**

---

## Project Structure

- **`index.html`** - The user interface layout, settings sliders, and displays.
- **`styles.css`** - Styling for the glassmorphism dark-mode theme.
- **`physics.js`** - Physical math equations for Raman gain, EDFA metastable fractions, and noise figures.
- **`nn.js`** - Feedforward neural network class (`SimpleNN`) and backpropagation written from scratch in pure JavaScript.
- **`visualizer.js`** - Canvas code to draw the 2D Fitness map and the Neural Network node structure.
- **`app.js`** - The central logic file that binds the sliders, buttons, charts, and optimization loop together.