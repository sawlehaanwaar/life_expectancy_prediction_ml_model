from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

from scripts.data_preprocessing_script import preprocess_data

def train_and_evaluate_model(data_source, alpha=0.0001):
    # Step 1: Load and preprocess the data
    df = preprocess_data(data_source)
    
    # Step 2: Split data into features (X) and target (y)
    X = df.drop('life expectancy', axis=1)
    y = df['life expectancy']
    
    # Step 3: Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 4: Train the Lasso regression model
    lasso_model = Lasso(alpha=alpha, random_state=42)
    lasso_model.fit(X_train, y_train)
    
    # Step 5: Make predictions on the test set
    y_pred = lasso_model.predict(X_test)
    
    # Step 6: Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")
    
    # Save the model
    joblib.dump(lasso_model, 'lasso_model.pkl')

    return lasso_model
