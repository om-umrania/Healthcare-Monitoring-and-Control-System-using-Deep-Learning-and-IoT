import pandas as pd
import numpy as np

# Define the number of samples
num_samples = 1000

# Generate synthetic data
synthetic_data = {
    'sweaty': np.random.randint(0, 2, num_samples),
    'high_temperature': np.random.randint(0, 2, num_samples),
    'normal_temperature': np.random.randint(0, 2, num_samples),
    'non_sweaty': np.random.randint(0, 2, num_samples),
    'itching_in_the_body': np.random.randint(0, 2, num_samples),
    'ECG_temp': np.random.uniform(0.7, 1.3, num_samples),
    'ECG_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'ECG_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'ECG_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'EEG_temp': np.random.uniform(0.7, 1.3, num_samples),
    'EEG_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'EEG_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'EEG_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'EMG_temp': np.random.uniform(0.7, 1.3, num_samples),
    'EMG_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'EMG_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'EMG_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'BP_temp': np.random.uniform(0.7, 1.3, num_samples),
    'BP_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'BP_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'BP_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'body_gas_temp': np.random.uniform(0.7, 1.3, num_samples),
    'body_gas_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'body_gas_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'body_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'oxygen_temp': np.random.uniform(0.7, 1.3, num_samples),
    'oxygen_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'oxygen_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'oxygen_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'glucose_temp': np.random.uniform(0.7, 1.3, num_samples),
    'glucose_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'glucose_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'glucose_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'kidney_temp': np.random.uniform(0.7, 1.3, num_samples),
    'kidney_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'kidney_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'kidney_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'liver_temp': np.random.uniform(0.7, 1.3, num_samples),
    'liver_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'liver_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'liver_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'facial_temp': np.random.uniform(0.7, 1.3, num_samples),
    'facial_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'facial_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'facial_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'skin_disease_temp': np.random.uniform(0.7, 1.3, num_samples),
    'skin_disease_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'skin_disease_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'skin_disease_gas_levels': np.random.uniform(0.95, 1.05, num_samples),
    'lung_temp': np.random.uniform(0.7, 1.3, num_samples),
    'lung_humidity': np.random.uniform(0.8, 1.2, num_samples),
    'lung_pressure': np.random.uniform(0.9, 1.1, num_samples),
    'lung_gas_levels': np.random.uniform(0.95, 1.05, num_samples)
}

# Create DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Save to CSV
synthetic_df.to_csv('synthetic_healthcare_data_new.csv', index=False)

print("Synthetic data generated and saved to 'synthetic_healthcare_data.csv'")