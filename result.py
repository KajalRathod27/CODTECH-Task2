import pandas as pd
import streamlit as st

# Load the saved evaluation results
results_df = pd.read_csv('model_evaluation_results.csv')

# Find the best model based on F1 Score
best_model = results_df.sort_values(by='F1 Score', ascending=False).iloc[0]

# Streamlit UI
st.title("Model Evaluation Dashboard")
st.dataframe(results_df)  # Display the model evaluation results
st.write(f"Recommended Model: {best_model['Model']} with F1 Score: {best_model['F1 Score']:.2f}")
