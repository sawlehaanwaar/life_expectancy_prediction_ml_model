# model_validation.py

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scripts.data_preprocessing_script import preprocess_data  

def perform_grid_search(data_source):

    # Loading and preprocessing the data
    df = preprocess_data(data_source)
    
    # Splitting data into features (X) and target (y)
    X = df.drop('life expectancy', axis=1)
    y = df['life expectancy']
    
    # Splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Defining parameter grid for Lasso model
    param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    
    # Initializing Lasso model
    lasso = Lasso(random_state=42)
    
    # Setting up GridSearchCV
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, 
                               cv=5, scoring='neg_mean_squared_error')
    
    # Fitting the GridSearchCV on the training data
    grid_search.fit(X_train, y_train)
    
    # Best parameters and best score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score (MSE): ", -grid_search.best_score_)
    
    # Best model and make predictions on the test set
    best_lasso = grid_search.best_estimator_
    y_pred = best_lasso.predict(X_test)
    
    # Evaluating the model on the test set
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test Mean Absolute Error: {mae}")
    print(f"Test Mean Squared Error: {mse}")
    print(f"Test Root Mean Squared Error: {rmse}")
    print(f"Test R-squared: {r2}")
    
    return best_lasso  