Simulation and Classification of Driver Behavior
This repository implements a framework for calibrating the Intelligent Driver Model (IDM) and classifying driving styles using high-resolution trajectory data from the highD dataset.

Project Overview
The system processes drone-captured vehicle trajectories to extract longitudinal dynamics and optimize behavioral parameters. It transitions from raw time-series data to distinct driver behavior profiles.

Key Features

IDM Calibration: Optimization of driver-specific parameters (v0,T,a,b,s ) using scipy-based optimization.

Data Processing: Extraction of velocity (v), acceleration (X_acc), and distance headway (s) from .pkl trajectory files.

Behavioral Classification: Categorization of drivers based on calibrated parameter distributions.

Visualization: Statistical plotting of parameter convergence and classification clusters.

Technical Stack
Language: Python 3.x

Libraries: pandas, numpy, scipy, scikit-learn, matplotlib

File Structure
script.py: Core optimization engine for IDM parameter calibration.

classification.py: Driver behavior categorization logic.

graph.py: Comparative visualization of V_IDM vs X_acc

idm_mean_parameters.csv: Aggregated behavioral data for clustering.

driver_classification.png: Visual output of the classification model.

Installation & Setup
1. Environment Requirements

Bash
pip install pandas numpy scipy scikit-learn matplotlib
2. Data Configuration

The scripts require the highD dataset structure. Ensure the following paths are set in your local environment:

META_DATA_PATH: Directory containing .csv metadata.

RAW_DATA_PATH: Path to the .pkl trajectory file (e.g., 241202_highD_lcTrainingTestForOptimisationData_di.pkl).

3. Execution Sequence

Calibration: Perform parameter optimization.

Bash
python script.py
Visualization: Generate comparative plots.

Bash
python graph.py
Classification: Run behavioral analysis.

Bash
python classification.py
Methodology
The input consists of drone recordings (25 FPS) covering 400m highway segments.

Input Features: Ego velocity (v), longitudinal acceleration (X_acc), and distance headway (s).

Pre-processing: Relative speed (Δv) calculation. Default values (s=250m,Δv=1.0m/s) are applied for free-flow traffic consistency.

Optimization: Vehicle class (Car/Truck) constraints are used to bound the parameter search space.