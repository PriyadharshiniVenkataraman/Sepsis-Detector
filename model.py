from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# Load the dataset
df = pd.read_csv("Dataset.csv")

# Drop rows with missing SepsisLabel (if any)
df = df.dropna(subset=['SepsisLabel'])

print("✅ After dropping rows with missing SepsisLabel:")
print(df.info())

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("\n✅ After imputing missing values:")
print(df_imputed.isnull().sum())  # Should show 0 for all

# Normalize selected vitals
vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
scaler = MinMaxScaler()
df_imputed[vitals] = scaler.fit_transform(df_imputed[vitals])

print("\n✅ After normalization:")
print(df_imputed[vitals].describe())  # Values should be between 0 and 1

# Define features and label
X = df_imputed.drop(columns=['SepsisLabel'])
y = df_imputed['SepsisLabel']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n✅ Training set shape: {X_train.shape}")
print(f"✅ Testing set shape: {X_test.shape}") 

#training
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Drop unnecessary columns like identifiers
X = df_imputed.drop(columns=['SepsisLabel', 'Unnamed: 0', 'Patient_ID'])

# Target variable
y = df_imputed['SepsisLabel']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
