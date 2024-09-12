# Importing necessary modules
from scripts.data_loading import load_data
from scripts.data_preprocessing_script import preprocess_data
from scripts.model_training_and_evaluation import train_and_evaluate_model
from scripts.model_validation import perform_grid_search
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"Test Mean Absolute Error: {mae}")
    print(f"Test Mean Squared Error: {mse}")
    print(f"Test Root Mean Squared Error: {rmse}")
    print(f"Test R-squared: {r2}")

def main():
    data_source = '/Users/sawlehaanwaar/Documents/GitHub/life_expectancy_prediction_ml_model/data/raw/life_expectancy_data.csv'

    df_processed = preprocess_data(data_source)

    print("\n--- Performing Grid Search for Hyperparameter Tuning ---")
    best_lasso_model, scaler, X_test_scaled, y_test, _  = perform_grid_search(df_processed)

    print("\n--- Evaluating the Best Model on the Test Data ---")
    evaluate_model(best_lasso_model, X_test_scaled, y_test)

if __name__ == "__main__":  
    main()
