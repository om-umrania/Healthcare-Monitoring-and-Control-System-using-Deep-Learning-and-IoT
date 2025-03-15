import pandas as pd
import numpy as np
import random

# Define number of samples
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
df = pd.DataFrame(data, columns=columns)

# Add Patient ID
df["patient_id"] = range(1, len(df) + 1)

# Rename environmental parameters for clarity
df.rename(
    columns={"lung_temp": "ambient_temperature", "facial_temp": "surface_temperature"},
    inplace=True
)

# Compute dependent internal parameters
df["T1"] = df["ECG_temp"] * np.random.uniform(0.98, 1.02)
df["T2"] = df["EEG_temp"] * np.random.uniform(0.98, 1.02)
df["O1"] = df["oxygen_pressure"] * np.random.uniform(0.98, 1.02)
df["O2"] = df["oxygen_pressure"] * np.random.uniform(0.98, 1.02)
df["H1"] = df["BP_humidity"] * np.random.uniform(0.98, 1.02)
df["H2"] = df["BP_humidity"] * np.random.uniform(0.98, 1.02)
df["GL1"] = df["glucose_temp"] * np.random.uniform(0.98, 1.02)
df["GL2"] = df["glucose_temp"] * np.random.uniform(0.98, 1.02)

# Save synthetic data
df.to_csv("synthetic_patient_data.csv", index=False)
print("Synthetic Data Generated")

# Load the dataset
df = pd.read_csv("synthetic_patient_data.csv")

# Algorithm parameters
Ni = 100   # Number of iterations
Ns = 50    # Number of solutions
Lf = 0.01  # Learning factor
Lmax = 10  # Maximum bed locations

# Initialize solutions
def initialize_solutions(Ns):
    return [{"bed_location": None, "fitness": 0.0, "checked": False} for _ in range(Ns)]

# Disease-Specific Fitness Functions
def calculate_fitness_general(row):
    internal_factor = (row["ECG_temp"] + row["EEG_temp"] + row["EMG_temp"]) / 3
    external_factor = (row["ambient_temperature"] + row["surface_temperature"]) / 2
    return (internal_factor / external_factor) if external_factor != 0 else 0

def calculate_fitness_covid(row):
    respiratory_factor = (row["O2"] - row["O1"]) if row["O2"] != 0 else 0
    temperature_factor = (row["surface_temperature"] - row["ambient_temperature"])
    return respiratory_factor * temperature_factor

def calculate_fitness_heart(row):
    cardiac_stress = (row["ECG_temp"] - row["BP_temp"]) if row["ECG_temp"] != 0 else 0
    oxygen_dependency = (row["O2"] - row["O1"]) if row["O2"] != 0 else 0
    return cardiac_stress * oxygen_dependency

def calculate_fitness_diabetes(row):
    glucose_stability = (row["GL2"] - row["GL1"]) if row["GL2"] != 0 else 0
    hydration_factor = (row["H2"] - row["H1"]) if row["H2"] != 0 else 0
    return glucose_stability * hydration_factor

# Apply fitness calculations
df["general_fitness_value"] = df.apply(calculate_fitness_general, axis=1)
df["covid_fitness_value"] = df.apply(calculate_fitness_covid, axis=1)
df["heart_fitness_value"] = df.apply(calculate_fitness_heart, axis=1)
df["diabetes_fitness_value"] = df.apply(calculate_fitness_diabetes, axis=1)

# Compute fitness thresholds
thresholds = {
    "general": df["general_fitness_value"].mean(),
    "covid": df["covid_fitness_value"].mean(),
    "heart": df["heart_fitness_value"].mean(),
    "diabetes": df["diabetes_fitness_value"].mean()
}

# Function to compute time to recover
def time_to_recover(fitness_val):
    return int(max(1, 100 / (fitness_val * 1000))) if fitness_val > 0 else 30

# Function to select the best patient location and apply checked status
def select_best_solution(df, fitness_col, threshold):
    best_idx = df[fitness_col].idxmax()
    best_row = df.loc[best_idx]

    checked_status = best_row[fitness_col] >= threshold

    return {
        "patient_id": int(best_row["patient_id"]),
        "bed_location": random.randint(1, Lmax),
        "fitness_value": round(best_row[fitness_col], 6),
        "time_to_recover": time_to_recover(best_row[fitness_col]),
        "checked": checked_status
    }

# Find best solutions
best_general = select_best_solution(df, "general_fitness_value", thresholds["general"])
best_covid = select_best_solution(df, "covid_fitness_value", thresholds["covid"])
best_heart = select_best_solution(df, "heart_fitness_value", thresholds["heart"])
best_diabetes = select_best_solution(df, "diabetes_fitness_value", thresholds["diabetes"])

# Print formatted results
print("\nBest Solution for General Case:")
print(f"  Patient ID:         {best_general['patient_id']}")
print(f"  Bed Location:       {best_general['bed_location']}")
print(f"  General Fitness:    {best_general['fitness_value']}")
print(f"  Time to Recover:    {best_general['time_to_recover']} days")
print(f"  Checked Status:     {best_general['checked']}")

print("\nBest Solution for COVID:")
print(f"  Patient ID:         {best_covid['patient_id']}")
print(f"  Bed Location:       {best_covid['bed_location']}")
print(f"  COVID Fitness:      {best_covid['fitness_value']}")
print(f"  Time to Recover:    {best_covid['time_to_recover']} days")
print(f"  Checked Status:     {best_covid['checked']}")

print("\nBest Solution for Heart Disease:")
print(f"  Patient ID:         {best_heart['patient_id']}")
print(f"  Bed Location:       {best_heart['bed_location']}")
print(f"  Heart Fitness:      {best_heart['fitness_value']}")
print(f"  Time to Recover:    {best_heart['time_to_recover']} days")
print(f"  Checked Status:     {best_heart['checked']}")

print("\nBest Solution for Diabetes:")
print(f"  Patient ID:         {best_diabetes['patient_id']}")
print(f"  Bed Location:       {best_diabetes['bed_location']}")
print(f"  Diabetes Fitness:   {best_diabetes['fitness_value']}")
print(f"  Time to Recover:    {best_diabetes['time_to_recover']} days")
print(f"  Checked Status:     {best_diabetes['checked']}")

# Save updated dataset
df.to_csv("updated_patient_data.csv", index=False)
print("\nUpdated patient data saved successfully!")
