import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1) GENERATE SYNTHETIC DATA
# ----------------------------------------------------------
num_samples = 1000

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

# Generate synthetic data (mean=1, std=0.1)
data = np.random.normal(1, 0.1, size=(num_samples, len(columns)))
df = pd.DataFrame(data, columns=columns)

# Add Patient ID
df["patient_id"] = range(1, len(df) + 1)

# Rename environmental parameters for clarity
df.rename(
    columns={"lung_temp": "ambient_temperature", "facial_temp": "surface_temperature"},
    inplace=True
)

# Compute dependent internal parameters (example transformations)
df["T1"] = df["ECG_temp"] * np.random.uniform(0.98, 1.02)
df["T2"] = df["EEG_temp"] * np.random.uniform(0.98, 1.02)
df["O1"] = df["oxygen_pressure"] * np.random.uniform(0.98, 1.02)
df["O2"] = df["oxygen_pressure"] * np.random.uniform(0.98, 1.02)
df["H1"] = df["BP_humidity"] * np.random.uniform(0.98, 1.02)
df["H2"] = df["BP_humidity"] * np.random.uniform(0.98, 1.02)
df["GL1"] = df["glucose_temp"] * np.random.uniform(0.98, 1.02)
df["GL2"] = df["glucose_temp"] * np.random.uniform(0.98, 1.02)

# Save initial synthetic data
df.to_csv("synthetic_patient_data.csv", index=False)
print("Synthetic Data Generated.")

# ----------------------------------------------------------
# 2) LOAD DATA & DEFINE PARAMETERS
# ----------------------------------------------------------
df = pd.read_csv("synthetic_patient_data.csv")

Ni = 100   # Number of iterations
Ns = 50    # Number of solutions
Lf = 0.01  # Learning factor
Lmax = 10  # Maximum bed locations

# ----------------------------------------------------------
# 3) FITNESS CALCULATIONS
# ----------------------------------------------------------
# Clip negative fitness to zero, as an example.

def calculate_fitness_general(row):
    internal_factor = (row["ECG_temp"] + row["EEG_temp"] + row["EMG_temp"]) / 3
    external_factor = (row["ambient_temperature"] + row["surface_temperature"]) / 2
    val = (internal_factor / external_factor) if external_factor != 0 else 0
    return max(val, 0)

def calculate_fitness_covid(row):
    respiratory_factor = (row["O2"] - row["O1"])
    temperature_factor = (row["surface_temperature"] - row["ambient_temperature"])
    val = respiratory_factor * temperature_factor
    return max(val, 0)

def calculate_fitness_heart(row):
    cardiac_stress = (row["ECG_temp"] - row["BP_temp"])
    oxygen_dependency = (row["O2"] - row["O1"])
    val = cardiac_stress * oxygen_dependency
    return max(val, 0)

def calculate_fitness_diabetes(row):
    glucose_stability = (row["GL2"] - row["GL1"])
    hydration_factor  = (row["H2"] - row["H1"])
    val = glucose_stability * hydration_factor
    return max(val, 0)

# Apply fitness calculations
df["general_fitness_value"]  = df.apply(calculate_fitness_general,  axis=1)
df["covid_fitness_value"]    = df.apply(calculate_fitness_covid,    axis=1)
df["heart_fitness_value"]    = df.apply(calculate_fitness_heart,    axis=1)
df["diabetes_fitness_value"] = df.apply(calculate_fitness_diabetes, axis=1)

# Compute thresholds (using means as an example)
thresholds = {
    "general":  df["general_fitness_value"].mean(),
    "covid":    df["covid_fitness_value"].mean(),
    "heart":    df["heart_fitness_value"].mean(),
    "diabetes": df["diabetes_fitness_value"].mean()
}

# ----------------------------------------------------------
# 4) SELECT BEST SOLUTIONS
# ----------------------------------------------------------
def time_to_recover(fitness_val):
    # Example function: the higher the fitness, the fewer days to recover
    return int(max(1, 100 / (fitness_val * 1000))) if fitness_val > 0 else 30

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

best_general  = select_best_solution(df, "general_fitness_value",  thresholds["general"])
best_covid    = select_best_solution(df, "covid_fitness_value",    thresholds["covid"])
best_heart    = select_best_solution(df, "heart_fitness_value",    thresholds["heart"])
best_diabetes = select_best_solution(df, "diabetes_fitness_value", thresholds["diabetes"])

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

# ----------------------------------------------------------
# 5) CLASSIFICATION METRICS
# ----------------------------------------------------------
# Create synthetic "actual" labels (example).
df["actual_general"]  = np.random.choice([0,1], size=len(df))
df["actual_covid"]    = np.random.choice([0,1], size=len(df))
df["actual_heart"]    = np.random.choice([0,1], size=len(df))
df["actual_diabetes"] = np.random.choice([0,1], size=len(df))

# Predicted labels: 1 if fitness_value >= threshold, else 0
df["pred_general"]  = np.where(df["general_fitness_value"]  >= thresholds["general"],  1, 0)
df["pred_covid"]    = np.where(df["covid_fitness_value"]    >= thresholds["covid"],    1, 0)
df["pred_heart"]    = np.where(df["heart_fitness_value"]    >= thresholds["heart"],    1, 0)
df["pred_diabetes"] = np.where(df["diabetes_fitness_value"] >= thresholds["diabetes"], 1, 0)

def compute_metrics(actual, predicted):
    """
    Returns a dictionary of accuracy, precision, recall, and F1
    based on the given actual and predicted arrays.
    """
    TP = ((actual == 1) & (predicted == 1)).sum()
    TN = ((actual == 0) & (predicted == 0)).sum()
    FP = ((actual == 0) & (predicted == 1)).sum()
    FN = ((actual == 1) & (predicted == 0)).sum()

    total = TP + TN + FP + FN
    accuracy  = (TP + TN) / total if total != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0

    return {
        "accuracy":  round(accuracy, 3),
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3)
    }

# Compute metrics per category
metrics_general  = compute_metrics(df["actual_general"],  df["pred_general"])
metrics_covid    = compute_metrics(df["actual_covid"],    df["pred_covid"])
metrics_heart    = compute_metrics(df["actual_heart"],    df["pred_heart"])
metrics_diabetes = compute_metrics(df["actual_diabetes"], df["pred_diabetes"])

# Create table of results
comparison_table = pd.DataFrame([
    ["General",  metrics_general["accuracy"],  metrics_general["precision"],
     metrics_general["recall"], metrics_general["f1"]],
    ["COVID",    metrics_covid["accuracy"],    metrics_covid["precision"],
     metrics_covid["recall"],   metrics_covid["f1"]],
    ["Heart",    metrics_heart["accuracy"],    metrics_heart["precision"],
     metrics_heart["recall"],   metrics_heart["f1"]],
    ["Diabetes", metrics_diabetes["accuracy"], metrics_diabetes["precision"],
     metrics_diabetes["recall"], metrics_diabetes["f1"]]
],
    columns=["Disease", "Accuracy", "Precision", "Recall", "F1"])

print("\nComparison of Accuracy, Precision, Recall, and F1:")
print(comparison_table)

# ----------------------------------------------------------
# 6) SAVE UPDATED DATA
# ----------------------------------------------------------
df.to_csv("updated_patient_data.csv", index=False)
print("\nUpdated patient data saved successfully!")

# ---------------------------------------------------------------
# 7) SIMULATE EPOCHS TO SHOW METRIC IMPROVEMENT OVER TIME (PLOTTING)
# ---------------------------------------------------------------
# We'll preserve epoch=45
num_epochs = 45
epoch_list = list(range(1, num_epochs + 1))

# Suppose we start from baseline metrics and gradually improve them
base_accuracy  = 0.5
base_precision = 0.4
base_recall    = 0.3

accuracy_vals  = []
precision_vals = []
recall_vals    = []
f1_vals        = []

for epoch in epoch_list:
    # Simulate small random increments
    base_accuracy  += np.random.uniform(0.01, 0.03)
    base_precision += np.random.uniform(0.01, 0.02)
    base_recall    += np.random.uniform(0.01, 0.02)

    # Ensure they don't exceed 1.0
    current_accuracy  = min(base_accuracy, 1.0)
    current_precision = min(base_precision, 1.0)
    current_recall    = min(base_recall, 1.0)

    # Compute F1
    if (current_precision + current_recall) > 0:
        current_f1 = 2 * current_precision * current_recall / (current_precision + current_recall)
    else:
        current_f1 = 0.0

    accuracy_vals.append(current_accuracy)
    precision_vals.append(current_precision)
    recall_vals.append(current_recall)
    f1_vals.append(current_f1)

# Create a DataFrame for the epoch-based metrics
epoch_metrics_df = pd.DataFrame({
    "Epoch": epoch_list,
    "Accuracy": accuracy_vals,
    "Precision": precision_vals,
    "Recall": recall_vals,
    "F1": f1_vals
})

print("\nSimulated Epoch-by-Epoch Metrics:")
print(epoch_metrics_df)

# ----------------------------------------------------------
# 8) LINE CHART: METRICS OVER EPOCHS
# ----------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(epoch_metrics_df["Epoch"], epoch_metrics_df["Accuracy"],  marker='o', label="Accuracy")
plt.plot(epoch_metrics_df["Epoch"], epoch_metrics_df["Precision"], marker='^', label="Precision")
plt.plot(epoch_metrics_df["Epoch"], epoch_metrics_df["Recall"],    marker='s', label="Recall")
plt.plot(epoch_metrics_df["Epoch"], epoch_metrics_df["F1"],        marker='d', label="F1-Score")

plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Line Chart: Metrics Improvement Over 45 Epochs")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# 9) BAR CHART OF METRICS PER EPOCH (2x2 Subplots)
# ----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Bar Charts: Metrics Over 45 Epochs")

axes = axes.flatten()

axes[0].bar(epoch_metrics_df["Epoch"], epoch_metrics_df["Accuracy"], color='skyblue')
axes[0].set_title("Accuracy by Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")

axes[1].bar(epoch_metrics_df["Epoch"], epoch_metrics_df["Precision"], color='orange')
axes[1].set_title("Precision by Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Precision")

axes[2].bar(epoch_metrics_df["Epoch"], epoch_metrics_df["Recall"], color='green')
axes[2].set_title("Recall by Epoch")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Recall")

axes[3].bar(epoch_metrics_df["Epoch"], epoch_metrics_df["F1"], color='purple')
axes[3].set_title("F1-Score by Epoch")
axes[3].set_xlabel("Epoch")
axes[3].set_ylabel("F1")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ----------------------------------------------------------
# 10) BAR CHART: RECOVERY DURATION (MONTHS) COMPARISON
# ----------------------------------------------------------
# Example data for references (replace with real data if available)
diseases = ['General', 'COVID', 'Heart', 'Diabetes']
ref_11   = [8, 10, 12, 10]
ref_15   = [6,  9, 10,  9]
sys_model= [5,  7,  9,  8]

x = np.arange(len(diseases))
width = 0.25

plt.figure(figsize=(8, 5))
plt.bar(x - width, ref_11,   width, label='Ref-11',       color='blue')
plt.bar(x,         ref_15,   width, label='Ref-15',       color='orange')
plt.bar(x + width, sys_model,width, label='System Model', color='green')

plt.xticks(x, diseases)
plt.ylabel("Recovery Duration (months)")
plt.title("Recovery Duration Comparison per Disease")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
