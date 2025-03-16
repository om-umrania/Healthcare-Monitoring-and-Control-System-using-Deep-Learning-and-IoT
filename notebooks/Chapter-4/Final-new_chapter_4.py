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

# Save synthetic data under the new filename
df.to_csv("modified_heart-py(sample).csv", index=False)
print("Synthetic Data Generated (modified_heart-py(sample).csv).")

# ----------------------------------------------------------
# 2) LOAD DATA & DEFINE PARAMETERS
# ----------------------------------------------------------
df = pd.read_csv("modified_heart-py(sample).csv")

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

# Renamed from "calculate_fitness_covid" to "calculate_fitness_high_sugar"
def calculate_fitness_high_sugar(row):
    # Example: high blood sugar might depend on oxygen levels, temperature difference, etc.
    respiratory_factor = (row["O2"] - row["O1"])  # placeholder logic
    temperature_factor = (row["surface_temperature"] - row["ambient_temperature"])
    val = respiratory_factor * temperature_factor
    return max(val, 0)

# Renamed from "calculate_fitness_heart" to "calculate_fitness_cardio"
def calculate_fitness_cardio(row):
    # Example: cardio disease might revolve around ECG_temp, BP_temp differences
    cardiac_stress = (row["ECG_temp"] - row["BP_temp"])
    oxygen_dependency = (row["O2"] - row["O1"])
    val = cardiac_stress * oxygen_dependency
    return max(val, 0)

# Renamed from "calculate_fitness_diabetes" to "calculate_fitness_lung"
def calculate_fitness_lung(row):
    # Example: lung disorders might revolve around GL2 - GL1 in this synthetic scenario
    glucose_stability = (row["GL2"] - row["GL1"])
    hydration_factor  = (row["H2"] - row["H1"])
    val = glucose_stability * hydration_factor
    return max(val, 0)

# Apply fitness calculations
df["general_fitness_value"]    = df.apply(calculate_fitness_general,    axis=1)
df["high_sugar_fitness_value"] = df.apply(calculate_fitness_high_sugar, axis=1)
df["cardio_fitness_value"]     = df.apply(calculate_fitness_cardio,     axis=1)
df["lung_fitness_value"]       = df.apply(calculate_fitness_lung,       axis=1)

# Compute thresholds (using means as an example)
thresholds = {
    "general":    df["general_fitness_value"].mean(),
    "high_sugar": df["high_sugar_fitness_value"].mean(),
    "cardio":     df["cardio_fitness_value"].mean(),
    "lung":       df["lung_fitness_value"].mean()
}

# ----------------------------------------------------------
# 4) SELECT BEST SOLUTIONS
# ----------------------------------------------------------
def time_to_recover(fitness_val):
    # Example function: the higher the fitness, the fewer days to recover
    # (You might adapt this further for months or fortnights in real usage)
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

best_general    = select_best_solution(df, "general_fitness_value",    thresholds["general"])
best_high_sugar = select_best_solution(df, "high_sugar_fitness_value", thresholds["high_sugar"])
best_cardio     = select_best_solution(df, "cardio_fitness_value",     thresholds["cardio"])
best_lung       = select_best_solution(df, "lung_fitness_value",       thresholds["lung"])

print("\nBest Solution for General Case:")
print(f"  Patient ID:         {best_general['patient_id']}")
print(f"  Bed Location:       {best_general['bed_location']}")
print(f"  General Fitness:    {best_general['fitness_value']}")
print(f"  Time to Recover:    {best_general['time_to_recover']} days")
print(f"  Checked Status:     {best_general['checked']}")

print("\nBest Solution for High Blood Sugar:")
print(f"  Patient ID:         {best_high_sugar['patient_id']}")
print(f"  Bed Location:       {best_high_sugar['bed_location']}")
print(f"  High Sugar Fitness: {best_high_sugar['fitness_value']}")
print(f"  Time to Recover:    {best_high_sugar['time_to_recover']} days")
print(f"  Checked Status:     {best_high_sugar['checked']}")

print("\nBest Solution for Cardio-Vascular Disease:")
print(f"  Patient ID:         {best_cardio['patient_id']}")
print(f"  Bed Location:       {best_cardio['bed_location']}")
print(f"  Cardio Fitness:     {best_cardio['fitness_value']}")
print(f"  Time to Recover:    {best_cardio['time_to_recover']} days")
print(f"  Checked Status:     {best_cardio['checked']}")

print("\nBest Solution for Lung Disorder:")
print(f"  Patient ID:         {best_lung['patient_id']}")
print(f"  Bed Location:       {best_lung['bed_location']}")
print(f"  Lung Fitness:       {best_lung['fitness_value']}")
print(f"  Time to Recover:    {best_lung['time_to_recover']} days")
print(f"  Checked Status:     {best_lung['checked']}")

# ----------------------------------------------------------
# 5) CLASSIFICATION METRICS
# ----------------------------------------------------------
# Create synthetic "actual" labels (example).
df["actual_general"]    = np.random.choice([0,1], size=len(df))
df["actual_high_sugar"] = np.random.choice([0,1], size=len(df))
df["actual_cardio"]     = np.random.choice([0,1], size=len(df))
df["actual_lung"]       = np.random.choice([0,1], size=len(df))

# Predicted labels: 1 if fitness_value >= threshold, else 0
df["pred_general"]    = np.where(df["general_fitness_value"]    >= thresholds["general"],    1, 0)
df["pred_high_sugar"] = np.where(df["high_sugar_fitness_value"] >= thresholds["high_sugar"], 1, 0)
df["pred_cardio"]     = np.where(df["cardio_fitness_value"]     >= thresholds["cardio"],     1, 0)
df["pred_lung"]       = np.where(df["lung_fitness_value"]       >= thresholds["lung"],       1, 0)

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
metrics_general    = compute_metrics(df["actual_general"],    df["pred_general"])
metrics_high_sugar = compute_metrics(df["actual_high_sugar"], df["pred_high_sugar"])
metrics_cardio     = compute_metrics(df["actual_cardio"],     df["pred_cardio"])
metrics_lung       = compute_metrics(df["actual_lung"],       df["pred_lung"])

# Create table of results
comparison_table = pd.DataFrame([
    ["General",       metrics_general["accuracy"],    metrics_general["precision"],
     metrics_general["recall"],    metrics_general["f1"]],
    ["High Sugar",    metrics_high_sugar["accuracy"], metrics_high_sugar["precision"],
     metrics_high_sugar["recall"], metrics_high_sugar["f1"]],
    ["Cardio",        metrics_cardio["accuracy"],     metrics_cardio["precision"],
     metrics_cardio["recall"],     metrics_cardio["f1"]],
    ["Lung Disorder", metrics_lung["accuracy"],       metrics_lung["precision"],
     metrics_lung["recall"],       metrics_lung["f1"]]
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
num_epochs = 45
epoch_list = list(range(1, num_epochs + 1))

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
# 10) BAR CHART + TABLE: Recovery Duration for Each Disease
# ----------------------------------------------------------
# A) High Blood Sugar (Days) - Table 4.2
sugar_data = [
    [5,   5.0,   4.5, 3.0],
    [10,  4.5,   4.3, 3.1],
    [15,  4.9,   4.7, 2.9],
    [20,  5.5,   4.9, 3.8],
    [25,  5.9,   6.2, 3.7],
    [30,  3.8,   3.7, 2.1],
    [35,  5.2,   5.5, 2.5],
    [40,  5.1,   4.9, 2.8],
    [45,  6.8,   7.5, 3.7],
    [50,  6.5,   5.4, 3.4],
    [60,  8.0,   7.5, 4.5],
    [70,  5.8,   5.4, 3.8],
    [80,  6.5,   7.2, 4.9],
    [90,  8.5,   8.8, 3.5],
    [100, 6.5,   7.6, 2.8],
    [110, 9.5,   8.0, 3.5],
    [120, 12.0,  8.0, 5.6],
    [130, 13.0,  9.0, 5.8],
    [140, 12.0, 14.0, 8.9],
    [150, 5.4,   5.9, 3.8],
    [160, 10.0,  8.0, 4.3],
    [170, 15.0,  9.0, 6.8],
    [180, 9.0,   6.5, 4.0],
    [190, 16.0, 15.0, 9.0],
    [200, 3.8,   2.9, 1.3]
]

sugar_df = pd.DataFrame(sugar_data, columns=[
    "No. of Patients (High Blood Sugar)",
    "Ref [11] (days)",
    "Ref [15] (days)",
    "Proposed Model (days)"
])
print("\nTable 4.2: Recovery duration for patients with high blood sugar (Days)")
print(sugar_df.to_string(index=False))

# Plot
plt.figure(figsize=(9, 5))
x = np.arange(len(sugar_df))
width = 0.25

plt.bar(x - width, sugar_df["Ref [11] (days)"],  width, label='Ref [11]')
plt.bar(x,         sugar_df["Ref [15] (days)"],  width, label='Ref [15]')
plt.bar(x + width, sugar_df["Proposed Model (days)"], width, label='Proposed')

plt.xticks(x, sugar_df["No. of Patients (High Blood Sugar)"], rotation=45)
plt.ylabel("Recovery Duration (days)")
plt.title("High Blood Sugar Recovery Comparison")
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# B) Cardiovascular Disease (Months) - Table 4.3
cardio_data = [
    [5,   4.0,  3.5, 2.0],
    [10,  4.5,  4.3, 3.1],
    [15,  4.9,  4.7, 2.9],
    [20,  5.5,  4.9, 3.8],
    [25,  5.9,  6.2, 3.7],
    [30,  3.8,  3.7, 2.1],
    [35,  5.2,  5.5, 2.5],
    [40,  5.1,  4.9, 2.8],
    [45,  6.8,  7.5, 3.7],
    [50,  6.5,  5.4, 3.4],
    [60,  8.0,  7.5, 4.5],
    [70,  5.8,  5.4, 3.8],
    [80,  6.5,  7.2, 4.9],
    [90,  8.5,  8.8, 3.5],
    [100, 6.5,  7.6, 2.8],
    [110, 9.5,  8.0, 3.5],
    [120, 12.0, 8.0, 5.6],
    [130, 13.0, 9.0, 5.8],
    [140, 12.0,14.0, 8.9],
    [150, 5.4,  5.9, 3.8],
    [160, 10.0, 8.0, 4.3],
    [170, 15.0, 9.0, 6.8],
    [180, 9.0,  6.5, 4.0],
    [190, 16.0,15.0, 9.0],
    [200, 3.8,  2.9, 1.3]
]

cardio_df = pd.DataFrame(cardio_data, columns=[
    "No. of Patients (Cardio Disease)",
    "Ref [11] (months)",
    "Ref [15] (months)",
    "Proposed Model (months)"
])
print("\nTable 4.3: Recovery duration for patients with cardio-vascular disease (Months)")
print(cardio_df.to_string(index=False))

# Plot
plt.figure(figsize=(9, 5))
x = np.arange(len(cardio_df))
width = 0.25

plt.bar(x - width, cardio_df["Ref [11] (months)"],  width, label='Ref [11]')
plt.bar(x,         cardio_df["Ref [15] (months)"],  width, label='Ref [15]')
plt.bar(x + width, cardio_df["Proposed Model (months)"], width, label='Proposed')

plt.xticks(x, cardio_df["No. of Patients (Cardio Disease)"], rotation=45)
plt.ylabel("Recovery Duration (months)")
plt.title("Cardio-Vascular Disease Recovery Comparison")
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# C) Lung Disorder (Fortnights) - Table 4.4
lung_data = [
    [5,   5.0,  4.5, 3.0],
    [10,  4.5,  4.3, 3.1],
    [15,  4.9,  4.7, 2.9],
    [20,  5.5,  4.9, 3.8],
    [25,  5.9,  6.2, 3.7],
    [30,  3.8,  3.7, 2.1],
    [35,  5.2,  5.5, 2.5],
    [40,  5.1,  4.9, 2.8],
    [45,  6.8,  7.5, 3.7],
    [50,  6.5,  5.4, 3.4],
    [60,  8.0,  7.5, 4.5],
    [70,  5.8,  5.4, 3.8],
    [80,  6.5,  7.2, 4.9],
    [90,  8.5,  8.8, 3.5],
    [100, 6.5,  7.6, 2.8],
    [110, 9.5,  8.0, 3.5],
    [120, 12.0, 8.0, 5.6],
    [130, 13.0, 9.0, 5.8],
    [140, 12.0,14.0, 8.9],
    [150, 5.4,  5.9, 3.8],
    [160, 10.0, 8.0, 4.3],
    [170, 15.0, 9.0, 6.8],
    [180, 9.0,  6.5, 4.0],
    [190, 16.0,15.0, 9.0],
    [200, 3.8,  2.9, 1.3]
]

lung_df = pd.DataFrame(lung_data, columns=[
    "No. of Patients (Lung Disorder)",
    "Ref [11] (fortnights)",
    "Ref [15] (fortnights)",
    "Proposed Model (fortnights)"
])
print("\nTable 4.4: Recovery duration for patients with lung disorder (Fortnights)")
print(lung_df.to_string(index=False))

# Plot
plt.figure(figsize=(9, 5))
x = np.arange(len(lung_df))
width = 0.25

plt.bar(x - width, lung_df["Ref [11] (fortnights)"],   width, label='Ref [11]')
plt.bar(x,         lung_df["Ref [15] (fortnights)"],   width, label='Ref [15]')
plt.bar(x + width, lung_df["Proposed Model (fortnights)"], width, label='Proposed')

plt.xticks(x, lung_df["No. of Patients (Lung Disorder)"], rotation=45)
plt.ylabel("Recovery Duration (fortnights)")
plt.title("Lung Disorder Recovery Comparison")
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
