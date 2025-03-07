import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
data = pd.read_csv('cleaned_file.csv')

# Drop non-numeric and irrelevant columns
data = data.drop(columns=['City', 'Date'], errors='ignore')

# Remove rows where target 'AQI_Bucket' is missing
data = data.dropna(subset=['AQI_Bucket'])

# Define features and target
features = data.drop(columns=['AQI','AQI_Bucket'], errors='ignore')
target = data['AQI_Bucket']

# Feature scaling
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Building the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model evaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Save the trained model and scaler as .pkl files
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Download files from Colab
from google.colab import files
files.download('rf_model.pkl')
files.download('scaler.pkl')
