import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# Load dataset
print(" Loading dataset...")
df = pd.read_csv('Dataset.csv')
df = df.dropna(subset=['SepsisLabel'])

#  Feature selection
features = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
    'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium',
    'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI',
    'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
    'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'
]
target = 'SepsisLabel'

# Clean data
df = df[features + [target]]
df.fillna(df.mean(numeric_only=True), inplace=True)

print(f" Dataset shape: {df.shape}")

#  Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
X = pd.DataFrame(X_scaled, columns=features)
y = df[target].astype(int)


#  Handle imbalance
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("After SMOTE - Class counts:", np.bincount(y_resampled))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train model
print(" Training XGBoostClassifier...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

#  Evaluation
y_pred = model.predict(X_test)
print("\n Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#Predict on new input
while True:
    print("\n Enter new patient data (or type 'exit' to quit):")
    new_data = []
    for feature in features:
        val = input(f"➡️ {feature}: ")
        if val.lower() == 'exit':
            print(" Exiting prediction loop.")
            exit()
        try:
            new_data.append(float(val))
        except ValueError:
            print("⚠️ Invalid input. Please enter a numeric value.")
            break
    else:
        # Apply same scaling
        new_data_scaled = scaler.transform([new_data])
        proba = model.predict_proba(new_data_scaled)[0][1]  # Probability of class 1 (sepsis)
        prediction = model.predict(new_data_scaled)

        print(f"\n🔍 Probability of Sepsis: {proba:.2f}")

        if proba >= 0.8:
            risk = "🔴 High Risk ⚠️"
        elif proba >= 0.5:
            risk = "🟡 Moderate Risk"
        else:
            risk = "🟢 Low Risk"

        if prediction[0] == 1:
            print(f"✅ Sepsis **PRESENT** | Risk Level: {risk}")
        else:
            print(f"❌ Sepsis **NOT Present** | Risk Level: {risk}")

