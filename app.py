# Task 2: Model Evaluation and Comparison

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import logging

# Setup Logging
logging.basicConfig(filename='model_evaluation.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load Dataset
data = pd.read_csv('Dataset/mushrooms.csv')

# Verify dataset structure
print("Dataset Columns:", data.columns)

# Encode Categorical Features
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

# Splitting data into features and target
X = data.drop(columns=['class'])  # Features
y = data['class']  # Target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to Evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=50)
}

# Evaluation Metrics
evaluation_results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    result = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    evaluation_results.append(result)
    logging.info(f"{model_name}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1 Score={f1}")
    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(evaluation_results)
print("\nModel Evaluation Results:\n", results_df)

# Recommend the Best Model
best_model = results_df.sort_values(by='F1 Score', ascending=False).iloc[0]
print(f"\nRecommended Model: {best_model['Model']} with F1 Score: {best_model['F1 Score']:.2f}")
logging.info(f"Best Model: {best_model['Model']} with F1 Score: {best_model['F1 Score']:.2f}")

# Optional: Save Evaluation Results
results_df.to_csv('model_evaluation_results.csv', index=False)
