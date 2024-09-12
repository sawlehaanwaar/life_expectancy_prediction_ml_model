# life_expectancy_prediction_ml_model
Machine Learning model created for prediction    of Life expectancies of nations
This model predicts life expectancy using machine learning techniques. The model is built using Lasso regression and includes preprocessing, hyperparameter tuning, and evaluation steps.

The repository also contains a machine learning pipeline designed to predict life expectancy based on multiple health and socio-economic factors. The model is built using Lasso regression, and the project follows a module-based structure for data preprocessing, model training, hyperparameter tuning, and evaluation.

PROJECT STRUCTURE
├── notebooks/ │ 
    ├── data_loading.ipynb # Data loading and exploration │ 
    ├── data_preprocessing.ipynb # Data preprocessing and transformation │ 
    ├── model_training_and_evaluation.ipynb # Model training and evaluation │ 
    ├── model_validation.ipynb # Model validation and hyperparameter tuning │ 
    ├── model_visualization.ipynb # Model result visualization 
    
├── scripts/ │ 
    ├── data_loading.py # Script for loading data │ 
    ├── data_preprocessing_script.py # Script for preprocessing the data │ 
    ├── model_training_and_evaluation.py # Script for training and evaluating the model │ 
    ├── model_validation.py # Script for hyperparameter tuning │ 
    ├── model_visualization.py # Script for visualizing the results ├── models/ │ 
    ├── trained_model.pkl # Saved trained model ├── data/ │ 
    ├── life_expectancy_data.csv # Dataset used for training 
    ├── ml_pipeline.py # Complete machine learning pipeline script 
    
└── README.md # Project documentation

MODEL DETAILS   
Model: Lasso Regression
Evaluation Metrics:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R²)

SETUP AND INSTALLATION
1 **Clone the repository:**
2 **Install the required dependencies:**
Ensure you have Python 3.x installed. Install the required packages using `pip`:
3 **Prepare the dataset:**
Place the `life_expectancy_data.csv` dataset in the `data/` directory.

USAGE
1 Running the Complete Pipeline
To run the full machine learning pipeline for training and evaluating the model:

This will preprocess the data, train the model, and save the best model to `trained_model.pkl`.

2 Running Individual Notebooks
Explore the different stages of model construction in these Jupyter notebooks:
- Data Loading: `data_loading.ipynb`
- Data Preprocessing: `data_preprocessing.ipynb`
- Exploratory Data Analysis `exploratory_data_analysis.ipynb`
- Model Training and Evaluation: `model_training_and_evaluation.ipynb`
- Model Validation: `model_validation.ipynb`
- Model Visualization: `model_visualization.ipynb`
The model itself is also stored in this folder:
- lasso_model.pkl

3 Testing the Model on Unseen Data

You can load the trained model and make predictions on unseen data by running the following:

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('/Users/sawlehaanwaar/Documents/GitHub/life_expectancy_prediction_ml_model/notebooks/lasso_model.pkl')

# Load unseen data
unseen_data = pd.read_csv('datapathway/unseen_life_expectancy_data.csv')

# Make predictions
predictions = model.predict(unseen_data)

print(predictions)


RESULTS 
The model's performance metrics on the test data:

Mean Absolute Error: X.XX
Mean Squared Error: X.XX
Root Mean Squared Error: X.XX

VISUALIZATION
After model training, the following visualizations are available:
1 Predicted vs Actual Values: A scatter plot showing the actual vs predicted life expectancy.
2 Residual Plot: Visualizing residuals to assess the model's errors.
3 Feature Importance: Analysis of which features contributed the most in the Lasso model.

