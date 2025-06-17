import pandas as pd
from crewai.tools import tool
import os

@tool("Walmart Store Sales Data Loader")
def load_and_filter_data(store_id: int) -> str:
    """
    Loads Walmart sales data, filters it for a single store,
    saves the cleaned data to a CSV file, and returns the file path.
    """
    file_path = 'data/Walmart_Sales.xlsx'
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return "Error: The data file 'data/Walmart_Sales.xlsx' was not found."

    df['ds'] = pd.to_datetime(df['ds'])
    store_df = df[df['Store'] == store_id].copy()

    if store_df.empty:
        return f"Error: No data found for Store ID {store_id}."

    # Create an 'outputs' directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Define the output path for the cleaned data
    output_file_path = f'outputs/store_{store_id}_cleaned_data.csv'
    
    # Save the DataFrame to a CSV file
    store_df.to_csv(output_file_path, index=False)
    
    print(f"Successfully loaded and saved cleaned data for Store ID: {store_id} to {output_file_path}")
    
    # Return the path to the newly created file
    return output_file_path