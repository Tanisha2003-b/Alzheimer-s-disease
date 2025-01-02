import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
file_path = 'alzheimers_disease_data.csv'
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())
print(df.head())
df['EducationLevel'] = df['EducationLevel'].round().astype(int)
df['AlcoholConsumption'] = df['AlcoholConsumption'].round().astype(int)
df['PhysicalActivity'] = df['PhysicalActivity'].round().astype(int)
df['DietQuality'] = df['DietQuality'].round().astype(int)
df['SleepQuality'] = df['SleepQuality'].round().astype(int)
df['SystolicBP'] = df['SystolicBP'].round().astype(int)
df['DiastolicBP'] = df['DiastolicBP'].round().astype(int)
df['CholesterolTotal'] = df['CholesterolTotal'].round().astype(int)
df['CholesterolLDL'] = df['CholesterolLDL'].round().astype(int)
df['CholesterolHDL'] = df['CholesterolHDL'].round().astype(int)
df['CholesterolTriglycerides'] = df['CholesterolTriglycerides'].round().astype(int)
df['MMSE'] = df['MMSE'].round().astype(int)
# List of columns you want to delete
columns_to_drop = ['PatientID','Ethnicity', 'ADL','DoctorInCharge','FunctionalAssessment']  # Replace with actual column names
# Drop the columns
data = df.drop(columns=columns_to_drop)

# Display the first few rows to verify the changes
print(data.head())


# Define your feature columns and target variable
X = data.drop(columns=['Diagnosis'])  # Replace 'target_column' with the actual name of your target variable
y = data['Diagnosis']

# Split the dataset into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the splits to verify
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

smote = SMOTE(random_state=42)
X_Resampled, y_Resampled = smote.fit_resample(X_train, y_train)
print("Balanced training set shape:", X_Resampled.shape, y_Resampled.shape)
print("Class distribution in y_train_balanced:\n", y_Resampled.value_counts())
rfc= RandomForestClassifier(n_estimators=100, random_state=42)  
rfc.fit(X_Resampled, y_Resampled)
y_pred =rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(rfc, 'model.pkl')


