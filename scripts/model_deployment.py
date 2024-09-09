import sys
import os

# Add the root directory of the project to the system path before importing anything else
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Print sys.path to confirm the root path has been added
print("Updated sys.path:", sys.path)

from fastapi import FastAPI
import joblib
import pandas as pd
from scripts.data_preprocessing_script import preprocess_data 

app = FastAPI()

# Load the saved Lasso model using an absolute path
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lasso_model.pkl')

if os.path.exists(model_path):
    print("Model file found!")
else:
    print("Model file not found!")

# Load the saved Lasso model
model = joblib.load(model_path)  # <-- Use model_path here to load the model

@app.post("/predict/")
def predict(data: dict):
    # Convert the incoming data into a pandas DataFrame
    df = pd.DataFrame([data])

    # Apply the same preprocessing that was used during training
    df_processed = preprocess_data(df)  # Preprocess the incoming data

    # Use the model to make a prediction
    prediction = model.predict(df_processed)

    # Return the prediction in a dictionary format
    return {"prediction": prediction.tolist()}