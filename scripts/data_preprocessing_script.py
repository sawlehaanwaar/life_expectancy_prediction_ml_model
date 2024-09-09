import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from scipy.stats import mstats

# Remove extra spaces from column labels
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.replace('  ', ' ')
    return df

# Convert all column names to lowercase
def lowercase_column_names(df):
    df.columns = df.columns.str.lower()

    return df

# Encode categorical variables (Label Encode 'Country' and One Hot Encode 'Status')
def encode_categorical_variables(df):
    label_encoder = LabelEncoder()
    df['country'] = label_encoder.fit_transform(df['country'])
    
    df = pd.get_dummies(df, columns=['status'], drop_first=True)

    if 'status_Developing' in df.columns:
        df['status_Developing'] = df['status_Developing'].astype(int)
    
    return df


# Deal with missing values
def impute_missing_values(df):
    median_imputer = SimpleImputer(strategy='median')
    df[['gdp', 'population']] = median_imputer.fit_transform(df[['gdp', 'population']])
    
    mean_columns = ['life expectancy', 'adult mortality', 'alcohol', 'hepatitis b', 'bmi', 
                    'polio', 'total expenditure', 'diphtheria', 'thinness 1-19 years', 
                    'thinness 5-9 years', 'income composition of resources', 'schooling']
    
    mean_imputer = SimpleImputer(strategy='mean')
    df[mean_columns] = mean_imputer.fit_transform(df[mean_columns])
    
    return df

# Deal with skewed data
def transform_skewed_data(df):
    df['gdp'] = np.log1p(df['gdp'])  
    df['population'] = np.log1p(df['population'])  
    df['measles'] = np.log1p(df['measles'])  
    df['percentage expenditure'] = np.log1p(df['percentage expenditure'])  
    df['hiv/aids'] = np.log1p(df['hiv/aids'])
    
    df['infant deaths'] = np.sqrt(df['infant deaths'])  
    df['under-five deaths'] = np.sqrt(df['under-five deaths'])  
    df['alcohol'] = np.sqrt(df['alcohol'])
    
    max_value = df['hepatitis b'].max()
    df['reflected_hepatitis_b'] = max_value - df['hepatitis b']
    df['hepatitis b'] = np.log1p(df['reflected_hepatitis_b'])
    df.drop(columns='reflected_hepatitis_b', inplace=True)
    
    return df

# Cap outliers
def cap_outliers(df, lower_percentile=0.01, upper_percentile=0.99):
    for column in df.select_dtypes(include=[np.number]).columns:
        lower_bound = df[column].quantile(lower_percentile)
        upper_bound = df[column].quantile(upper_percentile)
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

# Winsorize Percentage expenditure column
def winsorize_percentage_expenditure(df):
    df['percentage expenditure'] = mstats.winsorize(df['percentage expenditure'], limits=[0.01, 0.01])
    return df

# Scale the data
def scale_data(df):
    scaler = RobustScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# Full preprocessing pipeline
def preprocess_data(data_source):
    
    # Step 1: Load the data
    df = pd.read_csv(data_source)
    
    # Step 2: Clean column names
    df = clean_column_names(df)

    # Step 3: Convert column names to lowercase
    df = lowercase_column_names(df)
    
    # Step 3: Encode categorical variables
    df = encode_categorical_variables(df)
    
    # Step 4: Impute missing values
    df = impute_missing_values(df)
    
    # Step 5: Handle data skewness
    df = transform_skewed_data(df)
    
    # Step 6: Cap outliers
    df = cap_outliers(df)
    
    # Step 7: Winsorize 'Percentage expenditure' column
    df = winsorize_percentage_expenditure(df)
    
    # Step 8: Scale the data
    df = scale_data(df)
    
    return df