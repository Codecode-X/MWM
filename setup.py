"""
Neural ODE for ICU Patient Monitoring - Step 1: Setup
Project: Predicting adverse events in ICU using continuous-time models
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Create project directory structure
def setup_project_structure():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results',
        'notebooks',
        'src'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("[OK] Project structure created")

# Create requirements.txt
def create_requirements():
    """Generate requirements.txt file"""
    requirements = """torch>=2.0.0
torchvision>=0.15.0
torchdiffeq>=0.2.3
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
swanlab[dashboard]>=0.10.3,<0.11
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print("[OK] requirements.txt created")

# Create README
def create_readme():
    """Generate project README"""
    readme = """# Neural ODE for ICU Patient Monitoring

## Project Overview
Predicting adverse events (mortality, sepsis) in ICU patients using Neural Ordinary Differential Equations (Neural ODEs) for continuous-time modeling of patient vitals.

## Key Features
- Handles irregular time series sampling
- Continuous-time patient state modeling
- Uncertainty quantification
- Multi-variate vital signs integration

## Dataset
Using MIMIC-III Demo dataset (publicly available)
- 100 ICU patients
- Time-series vitals: Heart Rate, Blood Pressure, SpO2, Temperature, etc.

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
```
.
├── data/
│   ├── raw/          # Raw MIMIC-III data
│   └── processed/    # Preprocessed data
├── src/              # Source code
├── models/           # Saved models
├── results/          # Outputs and plots
└── notebooks/        # Jupyter notebooks
```

## Steps
1. [DONE] Setup and data preparation
2. Data preprocessing and feature engineering
3. Neural ODE model implementation
4. Training pipeline
5. Evaluation and visualization
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print("[OK] README.md created")

# Download MIMIC-III demo data instructions
def data_download_instructions():
    """Print instructions for downloading MIMIC-III demo"""
    instructions = """
================================================================
                   DATA DOWNLOAD INSTRUCTIONS                   
================================================================

MIMIC-III Demo Dataset (No credentialing required):

Option 1 - Direct Download:
1. Visit: https://physionet.org/content/mimiciii-demo/1.4/
2. Download the following files to data/raw/:
   - ADMISSIONS.csv
   - PATIENTS.csv
   - ICUSTAYS.csv
   - CHARTEVENTS.csv
   
Option 2 - Using wget (Linux/Mac):
   cd data/raw
   wget -r -N -c -np https://physionet.org/files/mimiciii-demo/1.4/

Option 3 - For this demo, we'll create synthetic data:
   We'll generate realistic synthetic ICU data for demonstration.

================================================================
"""
    print(instructions)

# Generate synthetic ICU data for demo purposes
def generate_synthetic_data():
    """Generate synthetic ICU patient data for demonstration"""
    np.random.seed(42)
    
    # Parameters
    n_patients = 100
    max_hours = 72  # 3 days ICU stay
    
    data = []
    
    for patient_id in range(1, n_patients + 1):
        # Random ICU stay duration
        stay_hours = np.random.randint(24, max_hours + 1)
        
        # Random measurement intervals (irregular sampling)
        n_measurements = np.random.randint(20, stay_hours)
        time_points = sorted(np.random.uniform(0, stay_hours, n_measurements))
        
        # Patient baseline (some patients are sicker)
        is_adverse = np.random.random() < 0.3  # 30% adverse events
        baseline_risk = 1.5 if is_adverse else 1.0
        
        for t in time_points:
            # Simulate vital signs with trends
            trend = t / stay_hours  # Things may worsen over time
            
            # Heart Rate (60-100 normal, higher if adverse)
            hr = np.random.normal(80 + baseline_risk * 15 * trend, 10)
            
            # Systolic BP (90-140 normal)
            sbp = np.random.normal(120 - baseline_risk * 20 * trend, 15)
            
            # Diastolic BP (60-90 normal)
            dbp = np.random.normal(80 - baseline_risk * 10 * trend, 10)
            
            # SpO2 (95-100 normal, lower if adverse)
            spo2 = np.random.normal(97 - baseline_risk * 5 * trend, 2)
            spo2 = np.clip(spo2, 85, 100)
            
            # Temperature (36.5-37.5 normal)
            temp = np.random.normal(37 + baseline_risk * 1 * trend, 0.5)
            
            # Respiratory Rate (12-20 normal)
            rr = np.random.normal(16 + baseline_risk * 8 * trend, 3)
            
            data.append({
                'patient_id': patient_id,
                'time_hours': round(t, 2),
                'heart_rate': round(hr, 1),
                'sbp': round(sbp, 1),
                'dbp': round(dbp, 1),
                'spo2': round(spo2, 1),
                'temperature': round(temp, 2),
                'respiratory_rate': round(rr, 1),
                'adverse_event': int(is_adverse)  # 1 = mortality/sepsis
            })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('data/raw/synthetic_icu_data.csv', index=False)
    
    print(f"[OK] Generated synthetic ICU data: {len(df)} measurements from {n_patients} patients")
    print(f"  - Adverse events: {df.groupby('patient_id')['adverse_event'].first().sum()} patients")
    print(f"  - Time range: 0-{df['time_hours'].max():.1f} hours")
    print(f"  - Saved to: data/raw/synthetic_icu_data.csv")
    
    return df

# Main setup function
def main():
    print("="*60)
    print("NEURAL ODE ICU MONITORING - STEP 1: PROJECT SETUP")
    print("="*60)
    
    setup_project_structure()
    create_requirements()
    create_readme()
    data_download_instructions()
    
    print("\n" + "="*60)
    print("Generating synthetic ICU data for demonstration...")
    print("="*60 + "\n")
    
    df = generate_synthetic_data()
    
    # Quick data preview
    print("\nData Preview:")
    print(df.head(10))
    
    print("\n" + "="*60)
    print("STEP 1 COMPLETE - SUCCESS")
    print("="*60)
    print("\nNext: Run 'pip install -r requirements.txt'")
    print("Then we'll proceed to Step 2: Data Preprocessing")

if __name__ == "__main__":
    main()