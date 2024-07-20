import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 1000

# Define column names
columns = [
    "sweaty", "high_temperature", "normal_temperature", "non_sweaty", "itching_in_the_body",
    "ECG_temp", "ECG_humidity", "ECG_pressure", "ECG_gas_levels",
    "EEG_temp", "EEG_humidity", "EEG_pressure", "EEG_gas_levels",
    "EMG_temp", "EMG_humidity", "EMG_pressure", "EMG_gas_levels",
    "BP_temp", "BP_humidity", "BP_pressure", "BP_gas_levels",
    "body_gas_temp", "body_gas_humidity", "body_gas_pressure", "body_gas_levels",
    "oxygen_temp", "oxygen_humidity", "oxygen_pressure", "oxygen_gas_levels",
    "glucose_temp", "glucose_humidity", "glucose_pressure", "glucose_gas_levels",
    "kidney_temp", "kidney_humidity", "kidney_pressure", "kidney_gas_levels",
    "liver_temp", "liver_humidity", "liver_pressure", "liver_gas_levels",
    "facial_temp", "facial_humidity", "facial_pressure", "facial_gas_levels",
    "skin_disease_temp", "skin_disease_humidity", "skin_disease_pressure", "skin_disease_gas_levels",
    "lung_temp", "lung_humidity", "lung_pressure", "lung_gas_levels"
]

# Generate synthetic data
data = np.random.normal(1, 0.1, size=(num_samples, len(columns)))

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Add binary columns for certain symptoms
df['sweaty'] = np.random.randint(0, 2, num_samples)
df['high_temperature'] = np.random.randint(0, 2, num_samples)
df['normal_temperature'] = np.random.randint(0, 2, num_samples)
df['non_sweaty'] = np.random.randint(0, 2, num_samples)
df['itching_in_the_body'] = np.random.randint(0, 2, num_samples)

# Add columns for the required variables
df['T1'] = np.random.normal(98.6, 1.0, num_samples)
df['T2'] = np.random.normal(99.0, 1.0, num_samples)
df['O1'] = np.random.normal(95, 2, num_samples)
df['O2'] = np.random.normal(94, 2, num_samples)
df['Text1'] = np.random.normal(70, 5, num_samples)
df['Text2'] = np.random.normal(72, 5, num_samples)
df['H1'] = np.random.normal(50, 10, num_samples)
df['H2'] = np.random.normal(55, 10, num_samples)
df['GL1'] = np.random.normal(100, 10, num_samples)
df['GL2'] = np.random.normal(105, 10, num_samples)

# Save to CSV
df.to_csv('synthetic_patient_data.csv', index=False)

print("Synthetic data generated and saved to synthetic_patient_data.csv")
