import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress the UserWarning from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

def run_simulation(num_weeks_to_simulate=10, num_top_products=5, epsilon=0.2):
    """
    Final version: Simulates the closed-loop process for multiple products,
    incorporating an epsilon-greedy strategy for exploration.
    """
    print("--- STARTING FINAL SIMULATION WITH EXPLORATION ---")
    print(f"Exploration Rate (Epsilon): {epsilon*100}%")

    # 1. Load initial data and model
    df = pd.read_csv('augmented_processed_data.csv')
    historical_model = joblib.load('historical_model.pkl')
    
    # Identify top products
    top_products = df['item_description'].value_counts().nlargest(num_top_products).index.tolist()
    print(f"Simulating for top products: {top_products}")

    # Main simulation loop
    for week in range(1, num_weeks_to_simulate + 1):
        print(f"\n{'='*15} SIMULATING WEEK {week} {'='*15}")
        df['date'] = pd.to_datetime(df['date'])
        new_weekly_data = []

        # Inner loop for each top product
        for product_name in top_products:
            print(f"\n--- Simulating Product: {product_name} ---")

            product_to_simulate = df[df['item_description'] == product_name].sort_values(by='date').iloc[[-1]]
            X_product_base = product_to_simulate.drop(columns=[
                'date', 'supplier', 'item_code', 'item_description', 'warehouse_sales'
            ])

            # --- a) Find the "Best" Price (Exploitation Choice) ---
            best_price = 0
            max_profit = -1
            base_price = product_to_simulate['retail_sales'].values[0]
            price_candidates = [base_price * 0.9, base_price, base_price * 1.1]

            for price in price_candidates:
                X_product_temp = X_product_base.copy()
                X_product_temp['retail_sales'] = price
                mean_demand, _ = get_probabilistic_forecast(historical_model, X_product_temp)
                cost = base_price * 0.6
                profit = (price - cost) * mean_demand
                
                if profit > max_profit:
                    max_profit = profit
                    best_price = price
            
            # --- NEW: Epsilon-Greedy Exploration Step ---
            if np.random.rand() < epsilon:
                # EXPLORE: Choose a random price instead of the best one
                other_prices = [p for p in price_candidates if p != best_price]
                if not other_prices:
                    final_prescribed_price = best_price
                else:
                    final_prescribed_price = np.random.choice(other_prices)
                print(f"*** EXPLORATION STEP *** Chose random price: ${final_prescribed_price:.2f}")
            else:
                # EXPLOIT: Choose the best-known price
                final_prescribed_price = best_price
                print(f"*** EXPLOITATION STEP *** Chose optimal price: ${final_prescribed_price:.2f}")

            # --- b) SIMULATE SALES OUTCOME ---
            X_product_final = X_product_base.copy()
            X_product_final['retail_sales'] = final_prescribed_price
            final_mean, final_std = get_probabilistic_forecast(historical_model, X_product_final)
            simulated_sales = max(0, int(np.random.normal(loc=final_mean, scale=final_std)))
            print(f"Simulated Sales Outcome: {simulated_sales} units")

            # --- c) PREPARE NEW DATA ROW ---
            new_data_row = product_to_simulate.copy()
            new_data_row['warehouse_sales'] = simulated_sales
            new_data_row['retail_sales'] = final_prescribed_price
            new_data_row['date'] = df['date'].max() + pd.Timedelta(weeks=1) # Increment date consistently
            new_weekly_data.append(new_data_row)

        # --- d) AUGMENT DATASET ---
        df = pd.concat([df] + new_weekly_data, ignore_index=True)
        print(f"\nDataset augmented. New size: {df.shape[0]} rows.")

        # --- e) RETRAIN MODEL ---
        y = df['warehouse_sales']
        X = df.drop(columns=['date', 'supplier', 'item_code', 'item_description', 'warehouse_sales'])
        historical_model.fit(X, y)
        joblib.dump(historical_model, 'historical_model.pkl')
        print("Model has been retrained with new weekly data.")

    print(f"\n{'='*15} SIMULATION COMPLETE {'='*15}")

def get_probabilistic_forecast(model, data_row):
    predictions = [tree.predict(data_row) for tree in model.estimators_]
    return np.mean(predictions), np.std(predictions)

if __name__ == '__main__':
    run_simulation()