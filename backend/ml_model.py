# backend/ml_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
import os

# Define the paths to your datasets
BALANCED_SPEECH_DATASET_PATH = 'Balanced_Speech_Dataset.csv'
MODEL_PATH = 'model.pkl'

def train_and_save_model():
    """
    Loads dataset and trains RFC model and saves model with symptoms and disorder names.
    """
    print(f"Loading data from {BALANCED_SPEECH_DATASET_PATH}...")
    try:
        df = pd.read_csv(BALANCED_SPEECH_DATASET_PATH)
        print("Data loaded successfully.")
        print("Dataset head:")
        print(df.head())
    except FileNotFoundError:
        print(f"Error: {BALANCED_SPEECH_DATASET_PATH} not found. Make sure it's in the backend directory.")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Identify symptom columns and the target column
    # Symptoms are all columns except Patient_ID, Age, Gender, and Disorder
    feature_columns = [col for col in df.columns if col not in ['Patient_ID', 'Age', 'Gender', 'Disorder']]
    target_column = 'Disorder'

    if not feature_columns:
        print("Error: No symptom columns found after excluding Patient_ID, Age, Gender, Disorder.")
        return
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        return

    # Store the ordered list of symptom names for consistent feature vector creation
    symptom_names = feature_columns
    print(f"\nIdentified Symptoms: {symptom_names}")

    # Prepare data for ML
    X = df[symptom_names]
    y = df[target_column]

    # Get unique disorder names and store their order
    disorder_names = sorted(y.unique().tolist()) # Sort for consistent mapping
    print(f"Identified Disorders: {disorder_names}")

    # Convert disorder names to numerical labels aka Label Encoding
    disorder_to_int = {name: i for i, name in enumerate(disorder_names)}
    y_encoded = y.map(disorder_to_int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier model
    print("Training RandomForestClassifier model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=disorder_names, zero_division=0))

    # Save the model and associated data
    model_data = {
        'model': model,
        'symptom_names': symptom_names, # Store the order of symptoms
        'disorder_names': disorder_names # Store the order of disorders for mapping predictions
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == '__main__':
    train_and_save_model()
