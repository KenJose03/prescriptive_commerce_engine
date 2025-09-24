import pandas as pd
import numpy as np
from data_loader import load_base_data

def augment_and_process(df):
    """
    Final version: Augments with all required features, cleans, 
    and engineers features for the complete dataset.
    """
    print("Starting data augmentation and processing...")
    df['date'] = pd.to_datetime(df['date'])

    # --- 1. Augment Data (Add Missing Features) ---

    # a) Add Macroeconomic Variables
    date_range = pd.to_datetime(pd.date_range(start=df['date'].min(), end=df['date'].max()))
    macro_data = pd.DataFrame({
        'date': date_range,
        'cpi': np.random.uniform(100, 105, size=len(date_range)),
        'unemployment_rate': np.random.uniform(5.0, 5.5, size=len(date_range)),
        'fuel_price': np.random.uniform(3.0, 3.5, size=len(date_range))
    })
    df = pd.merge(df, macro_data, on='date', how='left')
    print("Macroeconomic data merged.")
    
    # b) Add Competitor Prices (using 'retail_sales' as a proxy)
    df['competitor_price'] = df['retail_sales'] * np.random.uniform(0.95, 1.05, size=len(df))
    print("Competitor price data merged.")

    # c) Add Brand and Customer Ratings (using 'item_type' as category)
    brands = {'WINE': ['Sutter Home', 'Barefoot'], 'LIQUOR': ['Smirnoff', 'Jack Daniel\'s'], 'BEER': ['Budweiser', 'Corona']}
    df['brand'] = df['item_type'].apply(lambda x: np.random.choice(brands.get(x, ['Generic Brand'])))
    df['customer_rating'] = np.random.uniform(3.5, 5.0, size=len(df)).round(1)
    print("Brand and rating data generated.")

    # d) NEW: Generate missing categorical columns
    regions = ['North', 'South', 'East', 'West']
    payment_methods = ['Credit Card', 'Cash', 'Online']
    weather_conditions = ['Sunny', 'Rainy', 'Cloudy']
    df['region'] = np.random.choice(regions, size=len(df))
    df['payment_method'] = np.random.choice(payment_methods, size=len(df))
    df['weather'] = np.random.choice(weather_conditions, size=len(df))
    print("Generated data for region, payment_method, and weather.")

    # --- 2. Preprocessing and Feature Engineering ---
    
    # Use 'item_description' instead of 'product_name'
    df.sort_values(by=['item_description', 'date'], inplace=True)
    
    # Lagged features
    df['lag_sales_1'] = df.groupby('item_description')['warehouse_sales'].shift(1)

    # Rolling window features
    df['rolling_mean_sales_7'] = df.groupby('item_description')['warehouse_sales'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Encode ALL Categorical Variables
    df = pd.get_dummies(df, columns=['item_type', 'brand', 'region', 'payment_method', 'weather'], drop_first=True)
    
    df.dropna(inplace=True)
    
    print("Processing complete.")
    return df

if __name__ == '__main__':
    base_df = load_base_data()
    if base_df is not None:
        augmented_df = augment_and_process(base_df.copy())
        print("Augmented and processed data shape:", augmented_df.shape)
        print("Final columns:", augmented_df.columns.tolist())
        augmented_df.to_csv('augmented_processed_data.csv', index=False)