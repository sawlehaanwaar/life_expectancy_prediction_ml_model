from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
import numpy as np
import joblib
import os

# Importing preprocessing function
from scripts.data_preprocessing_script import preprocess_data

def train_and_evaluate_model(df_processed, alpha=0.01):

    # Splitting data into features (X) and target (y)
    X = df_processed.drop('life expectancy', axis=1)
    y = df_processed['life expectancy']
    
    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the training data set only to avoid data leakage and ensure patterns are learned by the model on the traininng data set
    # the same scaler is then used to transform the test data and new data after deployment

    #Initializing and fitting the scaler on training data set
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train) 

    #Transform test data set using same scaler
    X_test_scaled = scaler.transform(X_test) 

    # Saving the scaler to ensure we can use it after deployment on unseen dataset
    joblib.dump(scaler, 'scaler.pkl')

    # Training the Lasso regression model on the scaled data
    lasso_model = Lasso(alpha=alpha, random_state=42)
    lasso_model.fit(X_train_scaled, y_train)  # Fit model using scaled training data
    
    # Making predictions on the scaled test set
    y_pred = lasso_model.predict(X_test_scaled)
    
    # Evaluating the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")

    # Saving the model
    joblib.dump(lasso_model, 'lasso_model.pkl')

    return lasso_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred
