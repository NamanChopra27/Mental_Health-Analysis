import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from joblib import dump

# Load the dataset
data = pd.read_csv(r'C:\Users\naman\OneDrive\Desktop\Mental_Health\mental_health_data.csv.csv')

# Preprocess the 'What is your CGPA?' column
def preprocess_cgpa(cgpa):
    try:
        # Assuming 'What is your CGPA?' is a range, take the average
        return np.mean([float(val) for val in cgpa.split('-')])
    except (ValueError, AttributeError):
        # Handle non-numeric values or errors, you might want to customize this part
        return np.nan

data['What is your CGPA?'] = data['What is your CGPA?'].apply(preprocess_cgpa)

# Preprocess the data
le = LabelEncoder()
data['Depression'] = le.fit_transform(data['Do you have Depression?'])
data['Anxiety'] = le.fit_transform(data['Do you have Anxiety?'])
data['Panic_attack'] = le.fit_transform(data['Do you have Panic attack?'])

# Drop rows with missing values in specified columns
columns_with_missing_values = ['Age', 'What is your CGPA?', 'Depression', 'Anxiety', 'Panic_attack']
data = data.dropna(subset=columns_with_missing_values)

# Select features and target variable
features = ['Age', 'What is your CGPA?','Anxiety', 'Panic_attack']
X = data[features]
y = data['Depression']  # Replace 'Depression' with the actual column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
dump(model, 'mental_health_model.joblib')
