import sys
import os

# Adding the root directory of the project to the system path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Confirming root path has been added
print("Updated sys.path:", sys.path)

from fastapi import FastAPI
import joblib
import pandas as pd
from scripts.data_preprocessing_script import preprocess_data 

app = FastAPI()

# Loading the saved Lasso model using absolute path
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lasso_model.pkl')

if os.path.exists(model_path):
    print("Model file found!")
else:
    print("Model file not found!")

# Loading the saved Lasso model
model = joblib.load(model_path)  

@app.post("/predict/")
def predict(data: dict):
  
    df = pd.DataFrame([data])

    df_processed = preprocess_data(df)  

    prediction = model.predict(df_processed)

    return {"prediction": prediction.tolist()}