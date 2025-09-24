import pandas as pd

def load_base_data(filepath='Retail and wherehouse Sale.csv'):
    """Loads the dataset, standardizes column names, and creates a proper date column."""
    try:
        df = pd.read_csv(filepath)
        print("Base dataset loaded successfully.")

        # Standardize all column names to lowercase with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        print("Column names standardized.")

        # --- NEW: Create the 'date' column ---
        # Combine 'year' and 'month' to form a date string (assuming the 1st of the month)
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')

        # Drop the original year and month columns as they are now redundant
        df.drop(columns=['year', 'month'], inplace=True)

        print("A 'date' column has been successfully created.")
        print("Available columns:", df.columns.tolist())

        return df
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except KeyError as e:
        print(f"Error: A required column is missing from the CSV: {e}")
        return None

if __name__ == '__main__':
    base_df = load_base_data()
    if base_df is not None:
        print("\nFirst 5 rows of the loaded data:")
        print(base_df.head())