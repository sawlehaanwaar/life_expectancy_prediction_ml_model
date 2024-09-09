import pandas as pd

def load_data(data_source):

    try:
        df = pd.read_csv(data_source)
        
        return df
    
    except FileNotFoundError:
        print(f"Error: The file {data_source} was not found.")
        return None