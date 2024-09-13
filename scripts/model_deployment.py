from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
import sys

# Load trained Lasso model and scaler
model_path = '/Users/sawlehaanwaar/Documents/GitHub/life_expectancy_prediction_ml_model/scripts/best_lasso_model.pkl'
scaler_path = '/Users/sawlehaanwaar/Documents/GitHub/life_expectancy_prediction_ml_model/scripts/scaler.pkl'

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    logging.error(f"Error loading model or scaler: {str(e)}")
    sys.exit("Failed to load model or scaler")

# Define FastAPI app
app = FastAPI()

# Define the input data structure using raw column names
class LifeExpectancyInput(BaseModel):
    country: str
    year: int
    status: str
    adult_mortality: float
    infant_deaths: int
    alcohol: float
    percentage_expenditure: float
    hepatitis_B: float
    measles: int
    bmi: float
    under_five_deaths: int
    polio: float
    total_expenditure: float
    diphtheria: float
    hiv_aids: float
    gdp: float
    population: float
    thinness_1_19_years: float
    thinness_5_9_years: float
    income_composition_of_resources: float
    schooling: float

# Preprocess function to clean column names
def preprocess_data(df):
    # Clean the column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace('/', '_')
    
    # Further preprocessing steps (e.g., encoding, imputation, etc.)
    
    return df

# Function to restore original column names based on the mapping
def restore_original_column_names(df, column_mapping):
    return df.rename(columns=column_mapping)

@app.post("/predict/")
async def predict_life_expectancy(input_data: LifeExpectancyInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        logging.info(f"Input DataFrame: {input_df}")

        # Preprocess the data (clean column names)
        processed_df = preprocess_data(input_df)
        logging.info(f"Processed DataFrame: {processed_df}")

        # Ensure processed data has the correct columns
        if processed_df.shape[1] != scaler.n_features_in_:
            raise ValueError("Processed data does not have the correct number of features.")

        # Scale the input data
        input_scaled = scaler.transform(processed_df)
        logging.info(f"Scaled Data: {input_scaled}")

        # Make the prediction
        prediction = model.predict(input_scaled)
        logging.info(f"Prediction: {prediction}")

        # Return the prediction as a response
        return {"predicted_life_expectancy": prediction[0]}

    except ValueError as ve:
        logging.error(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction.")