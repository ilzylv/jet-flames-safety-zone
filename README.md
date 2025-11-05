# Thermal radiation safety zone analysis
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)]() 
[![Made with](https://img.shields.io/badge/made%20with-Python-green)]()
\
This repository contains the Python source code for the simulation and analysis presented in the paper "COMPARATIVE ANALYSIS AND MODELING OF SAFETY DISTANCES FOR THERMAL RADIATION EXPOSURE FROM METHANE JET FLAMES".
## Background
The primary objective of this project is to compare three distinct thermal radiation models (API 521, De Ris Adapted, and Caetano et al.) to determine safety distances for methane jet flames. The simulation quantifies the discrepancies between classical (power-based) models and recent empirical (data-driven) models.
\
## How to Run
1. Clone the repository: 
   ```bash
   git clone https://github.com/ilzylv/methane-jet-flame-analysis.git
   cd methane-jet-flame-analysis

2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
3. Run the main simulation script:
   ```bash
   python main.py
4. The results, graphs, and tables will be generated and saved to a new directory named outputs/. The final report will be printed to your terminal.

## Key References
**[1]** Caetano, N.R., et al. (2018). _Assessment of Thermal Radiation Heat Loss from Jet Diffusion Flames_. (Provides basis for radiative fractions).
\
**[2]** Caetano, N.R., et al. (2020). _Modelling Safety Distance from Industrial Turbulent Non-Premixed Gaseous Jet Flames_. (Used as the experimental baseline and for the empirical model $D = C_1 L_f + C_2$).
\
**[3]** Peng, Y., et al. (2025). Experimental Study on Combustion Characteristics of Methane Vertical Jet Flame. (Used for the effective flame temperature: 1051°C / 1324 K).
\
**[4]** NOAA. (2025). _Thermal Radiation Levels of Concern_. (Used to define the 5 critical heat flux levels for the safety zones).
\
**[5]** Johnson, D.M., et al. (2021). _Review of the Current Understanding of Hydrogen Jet Fires_. (Used to validate the flame length formula $L_f(Q_T)$).
\
**[6]** Sengupta, A., et al. (2018). _One Dimensional Modeling of Jet Diffusion Flame_. (Provides general background on flame modeling).