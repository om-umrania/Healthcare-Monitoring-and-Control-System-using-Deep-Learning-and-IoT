import pandas as pd
import numpy as np
import random

# Load the synthetic dataset
df = pd.read_csv('synthetic_patient_data.csv')

print(df.columns)


# Define the parameters for the algorithm
Ni = 100  # Number of iterations
Ns = 50   # Number of solutions
Lf = 0.01  # Learning factor
Lmax = 10  # Maximum number of locations (or beds) available

# Initialize solutions
def initialize_solutions(Ns):
    return [{"location": None, "fitness": 0, "changed": True} for _ in range(Ns)]

# Define fitness calculation functions for different cases
def calculate_fitness_general(Int1, Int2, Ext1, Ext2):
    f1 = (Int2 - Int1) / Int2
    f2 = (Ext2 - Ext1) / Ext2
    return f1 * f2

def calculate_fitness_covid(O1, O2, Text1, Text2):
    return ((T2 - T1) / T2) * ((O2 - O1) / O2) * ((Text2 - Text1) / Text2)

def calculate_fitness_heart(ECG1, ECG2, BP1, BP2, Text1, Text2, H1, H2):
    return ((ECG2 - ECG1) / ECG2) * ((BP2 - BP1) / BP2) * ((Text2 - Text1) / Text2) * ((H2 - H1) / H2)

def calculate_fitness_diabetes(GL1, GL2, O1, O2, Text1, Text2, H1, H2):
    return ((GL2 - GL1) / GL2) * ((O2 - O1) / O2) * ((Text2 - Text1) / Text2) * ((H2 - H1) / H2)

# Function to update patient location using the specified fitness calculation function

def update_patient_location(solutions, df, Ni, Ns, Lmax, Lf, fitness_function, param_columns):
    for iteration in range(Ni):
        for i in range(Ns):
            if solutions[i]["changed"]:
                # Randomly assign a new location
                solutions[i]["location"] = random.randint(1, Lmax)
                
                # Randomly select indices for the data
                indices = np.random.choice(df.index, 2, replace=False)
                
                # Extract parameters for the fitness function
                params = [df.loc[indices[j], param_columns[k]] for j in range(2) for k in range(len(param_columns) // 2)]
                
                # Calculate fitness value
                solutions[i]["fitness"] = fitness_function(*params)
        
        # Calculate average fitness and fitness threshold
        avg_fitness = np.mean([solution["fitness"] for solution in solutions])
        fitness_threshold = avg_fitness * Lf
        
        # Discard solutions below the threshold and mark others as changed
        for solution in solutions:
            if solution["fitness"] < fitness_threshold:
                solution["changed"] = False
            else:
                solution["changed"] = True
        
        # Select the solution with the maximum fitness
        best_solution = max(solutions, key=lambda x: x["fitness"])
        
        # Shift the patient to the location corresponding to the best solution
        for solution in solutions:
            if solution == best_solution:
                solution["location"] = best_solution["location"]
                solution["changed"] = False
                
    return best_solution

# Define parameters for each case
param_columns_general = ['ECG_temp', 'oxygen_temp', 'lung_temp', 'EEG_temp']
param_columns_covid = ['ECG_temp', 'oxygen_temp', 'lung_temp']
param_columns_heart = ['ECG_temp', 'BP_temp', 'lung_temp', 'lung_humidity']
param_columns_diabetes = ['glucose_temp', 'oxygen_temp', 'lung_temp', 'lung_humidity']

# Execute the algorithm for each case with synthetic data
solutions_general = initialize_solutions(Ns)
best_solution_general = update_patient_location(solutions_general, df, Ni, Ns, Lmax, Lf, calculate_fitness_general, param_columns_general)
print("Best Solution for General Case with synthetic data:", best_solution_general)

solutions_covid = initialize_solutions(Ns)
best_solution_covid = update_patient_location(solutions_covid, df, Ni, Ns, Lmax, Lf, calculate_fitness_covid, param_columns_covid)
print("Best Solution for COVID with synthetic data:", best_solution_covid)

solutions_heart = initialize_solutions(Ns)
best_solution_heart = update_patient_location(solutions_heart, df, Ni, Ns, Lmax, Lf, calculate_fitness_heart, param_columns_heart)
print("Best Solution for Heart Disease with synthetic data:", best_solution_heart)

solutions_diabetes = initialize_solutions(Ns)
best_solution_diabetes = update_patient_location(solutions_diabetes, df, Ni, Ns, Lmax, Lf, calculate_fitness_diabetes, param_columns_diabetes)
print("Best Solution for Diabetes with synthetic data:", best_solution_diabetes)