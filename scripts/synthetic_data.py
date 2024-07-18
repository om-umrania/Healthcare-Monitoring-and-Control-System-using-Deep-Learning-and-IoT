import pandas as pd
import numpy as np

# Define column names based on the provided datasets
columns = [
    'male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP',
    'BMI', 'heartRate', 'glucose', 'TenYearCHD', 'PatientId', 'Gender',
    'AirPollution', 'Alcohol use', 'DustAllergy', 'OccupationalHazards',
    'GeneticRisk', 'chronicLungDisease', 'BalancedDiet', 'Obesity', 'Smoking',
    'PassiveSmoker', 'ChestPain', 'CoughingofBlood', 'Fatigue', 'WeightLoss',
    'ShortnessofBreath', 'Wheezing', 'SwallowingDifficulty', 'ClubbingofFingerNails',
    'FrequentCold', 'DryCough', 'Snoring', 'Level', 'avgAnnCount', 'medIncome',
    'popEst2015', 'povertyPercent', 'binnedInc', 'MedianAge', 'MedianAgeMale',
    'MedianAgeFemale', 'Geography', 'AvgHouseholdSize', 'PercentMarried',
    'PctNoHS18_24', 'PctHS18_24', 'PctSomeCol18_24', 'PctBachDeg18_24',
    'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over',
    'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 'PctWhite',
    'PctBlack', 'PctAsian', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate',
    'deathRate', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'DiabetesPedigreeFunction', 'Outcome'
]

# Number of samples to generate
num_samples = 1000

# Generate synthetic data
synthetic_data = {
    'male': np.random.randint(0, 2, num_samples),
    'age': np.random.randint(18, 90, num_samples),
    'education': np.random.randint(1, 5, num_samples),
    'currentSmoker': np.random.randint(0, 2, num_samples),
    'cigsPerDay': np.random.randint(0, 40, num_samples),
    'BPMeds': np.random.randint(0, 2, num_samples),
    'prevalentStroke': np.random.randint(0, 2, num_samples),
    'prevalentHyp': np.random.randint(0, 2, num_samples),
    'diabetes': np.random.randint(0, 2, num_samples),
    'totChol': np.random.randint(150, 300, num_samples),
    'sysBP': np.random.randint(90, 200, num_samples),
    'diaBP': np.random.randint(60, 120, num_samples),
    'BMI': np.random.uniform(15, 40, num_samples),
    'heartRate': np.random.randint(50, 100, num_samples),
    'glucose': np.random.randint(70, 200, num_samples),
    'TenYearCHD': np.random.randint(0, 2, num_samples),
    'PatientId': np.arange(1, num_samples + 1),
    'Gender': np.random.choice(['Male', 'Female'], num_samples),
    'AirPollution': np.random.randint(1, 10, num_samples),
    'Alcohol use': np.random.randint(0, 2, num_samples),
    'DustAllergy': np.random.randint(0, 2, num_samples),
    'OccupationalHazards': np.random.randint(0, 2, num_samples),
    'GeneticRisk': np.random.randint(1, 10, num_samples),
    'chronicLungDisease': np.random.randint(0, 2, num_samples),
    'BalancedDiet': np.random.randint(0, 2, num_samples),
    'Obesity': np.random.randint(0, 2, num_samples),
    'Smoking': np.random.randint(0, 2, num_samples),
    'PassiveSmoker': np.random.randint(0, 2, num_samples),
    'ChestPain': np.random.randint(0, 2, num_samples),
    'CoughingofBlood': np.random.randint(0, 2, num_samples),
    'Fatigue': np.random.randint(0, 2, num_samples),
    'WeightLoss': np.random.randint(0, 2, num_samples),
    'ShortnessofBreath': np.random.randint(0, 2, num_samples),
    'Wheezing': np.random.randint(0, 2, num_samples),
    'SwallowingDifficulty': np.random.randint(0, 2, num_samples),
    'ClubbingofFingerNails': np.random.randint(0, 2, num_samples),
    'FrequentCold': np.random.randint(0, 2, num_samples),
    'DryCough': np.random.randint(0, 2, num_samples),
    'Snoring': np.random.randint(0, 2, num_samples),
    'Level': np.random.randint(1, 4, num_samples),
    'avgAnnCount': np.random.uniform(1000, 10000, num_samples),
    'medIncome': np.random.uniform(30000, 100000, num_samples),
    'popEst2015': np.random.randint(10000, 1000000, num_samples),
    'povertyPercent': np.random.uniform(0, 30, num_samples),
    'binnedInc': np.random.randint(1, 10, num_samples),
    'MedianAge': np.random.uniform(20, 60, num_samples),
    'MedianAgeMale': np.random.uniform(20, 60, num_samples),
    'MedianAgeFemale': np.random.uniform(20, 60, num_samples),
    'Geography': np.random.choice(['Urban', 'Rural'], num_samples),
    'AvgHouseholdSize': np.random.uniform(1, 5, num_samples),
    'PercentMarried': np.random.uniform(20, 80, num_samples),
    'PctNoHS18_24': np.random.uniform(0, 50, num_samples),
    'PctHS18_24': np.random.uniform(0, 50, num_samples),
    'PctSomeCol18_24': np.random.uniform(0, 50, num_samples),
    'PctBachDeg18_24': np.random.uniform(0, 50, num_samples),
    'PctHS25_Over': np.random.uniform(0, 100, num_samples),
    'PctBachDeg25_Over': np.random.uniform(0, 100, num_samples),
    'PctEmployed16_Over': np.random.uniform(0, 100, num_samples),
    'PctUnemployed16_Over': np.random.uniform(0, 20, num_samples),
    'PctPrivateCoverage': np.random.uniform(0, 100, num_samples),
    'PctEmpPrivCoverage': np.random.uniform(0, 100, num_samples),
    'PctPublicCoverage': np.random.uniform(0, 100, num_samples),
    'PctWhite': np.random.uniform(0, 100, num_samples),
    'PctBlack': np.random.uniform(0, 100, num_samples),
    'PctAsian': np.random.uniform(0, 100, num_samples),
    'PctOtherRace': np.random.uniform(0, 100, num_samples),
    'PctMarriedHouseholds': np.random.uniform(0, 100, num_samples),
    'BirthRate': np.random.uniform(0, 50, num_samples),
    'deathRate': np.random.uniform(0, 20, num_samples),
    'Pregnancies': np.random.randint(0, 15, num_samples),
    'Glucose': np.random.uniform(50, 200, num_samples),
    'BloodPressure': np.random.uniform(80, 200, num_samples),
    'SkinThickness': np.random.uniform(10, 50, num_samples),
    'Insulin': np.random.uniform(15, 276, num_samples),
    'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, num_samples),
    'Outcome': np.random.randint(0, 2, num_samples)
}

# Create DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Save to CSV
synthetic_df.to_csv('synthetic_healthcare_data.csv', index=False)

print("Synthetic data generated and saved to 'synthetic_healthcare_data.csv'")