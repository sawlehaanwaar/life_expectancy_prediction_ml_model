from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd

from .data_loading import load_data
from .data_preprocessing_script import preprocess_data
from .model_validation import perform_grid_search
from pydantic import BaseModel, Field


# Initialize FastAPI 
app = FastAPI()

# Load the trained model and scaler
model_path = os.path.join(os.path.dirname('/Users/sawlehaanwaar/Documents/GitHub/life_expectancy_prediction_ml_model/scripts/best_lasso_model.pkl'), 'best_lasso_model.pkl')
scaler_path = os.path.join(os.path.dirname('/Users/sawlehaanwaar/Documents/GitHub/life_expectancy_prediction_ml_model/scripts/scaler.pkl'), 'scaler.pkl')

# Attempt to load the model and scaler

model = joblib.load(model_path)
scaler = joblib.load(scaler_path )
print(f"Model and scaler loaded successfully.")

# Define the input data structure
class PredictionInput(BaseModel):
    country_encoded: int = Field(alias="country encoded")
    year: int
    status_developing: int = Field(alias="status developing")
    adult_mortality: float = Field(alias="adult mortality")
    infant_deaths: int = Field(alias="infant deaths")
    alcohol: float
    percentage_expenditure: float = Field(alias="percentage expenditure")
    hepatitis_b: float = Field(alias="hepatitis b")
    measles: int
    bmi: float
    under_five_deaths: int = Field(alias="under five deaths")
    polio: float
    total_expenditure: float = Field(alias="total expenditure")
    diphtheria: float
    hiv_aids: float = Field(alias="hiv/aids")
    gdp: float
    population: float
    thinness_1_19_years: float = Field(alias="thinness 1-19 years")
    thinness_5_9_years: float = Field(alias="thinness 5-9 years")
    income_composition_of_resources: float = Field(alias="income composition of resources")
    schooling: float

# Preprocessing function (modularize)
def preprocess_input(input_data: PredictionInput):

    try:
        data = np.array([[
            input_data.country_encoded,
            input_data.year,
            input_data.status_developing,
            input_data.adult_mortality,
            input_data.infant_deaths,
            np.sqrt(input_data.alcohol),
            np.log1p(input_data.percentage_expenditure), 
            np.sqrt(input_data.hepatitis_b), 
            np.log1p(input_data.measles),
            input_data.bmi,
            np.sqrt(input_data.under_five_deaths),
            input_data.polio,
            input_data.total_expenditure,
            input_data.diphtheria,
            np.log1p(input_data.hiv_aids),
            np.log1p(input_data.gdp), 
            np.log1p(input_data.population), 
            input_data.thinness_1_19_years,
            input_data.thinness_5_9_years,
            input_data.income_composition_of_resources,
            input_data.schooling
        ]])
        return data
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Preprocess input data
        data = preprocess_input(input_data)

        # Define the exact feature names that match your training data (with spaces and special characters)
        feature_names = [
            "Country", "Year", "Status", "Adult Mortality", "infant deaths", 
            "Alcohol", "percentage expenditure", "Hepatitis B", "Measles ", 
            " BMI ", "under-five deaths ", "Polio", "Total expenditure", 
            "Diphtheria ", " HIV/AIDS", "GDP", "Population", 
            " thinness  1-19 years", " thinness 5-9 years", 
            "Income composition of resources", "Schooling"
        ]

        # Create a mapping of the API input fields to the correct dataset feature names
        field_mapping = {
            "country encoded": "Country",
            "status developing": "Status",
            "under five deaths": "under-five deaths ",
            "adult mortality": "Adult Mortality",
            "infant deaths": "infant deaths",
            "alcohol": "Alcohol",
            "percentage expenditure": "percentage expenditure",
            "hepatitis b": "Hepatitis B",
            "measles": "Measles ",
            "bmi": " BMI ",
            "total expenditure": "Total expenditure",
            "diphtheria": "Diphtheria ",
            "hiv/aids": " HIV/AIDS",
            "gdp": "GDP",
            "population": "Population",
            "thinness 1-19 years": " thinness  1-19 years",
            "thinness 5-9 years": " thinness 5-9 years",
            "income_composition of resources": "Income composition of resources",
            "schooling": "Schooling"
        }

        # Replace the incoming field names with the exact feature names from the dataset
        data_renamed = pd.DataFrame(data, columns=[field_mapping.get(col, col) for col in feature_names])

        # Apply scaling (with the correct feature names)
        data_scaled = scaler.transform(data_renamed)

        # Make a prediction
        prediction = model.predict(data_scaled)

        # Debugging: Print the prediction value
        print(f"Prediction (before rounding): {prediction[0]}")

        # Return the prediction as a response
        return {"predicted_life_expectancy": round(prediction[0], 2)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
