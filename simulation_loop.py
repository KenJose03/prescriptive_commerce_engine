import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress the UserWarning from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)


def get_true_market_demand(prescribed_price, base_price, base_demand, market_conditions, shock_multiplier=1.0):
    """
    This is our "hidden" ground truth model. The ML model does not know the rules here.
    It simulates the real market's response to our pricing and market conditions.
    """
    # Effect of price deviation from base price (price elasticity)
    price_effect = (base_price - prescribed_price) * 2.5  # For every $1 cheaper, sell 2.5 more units

    # Effect of competitor price (cross-elasticity)
    competitor_effect = (market_conditions['competitor_price'] - prescribed_price) * 0.5

    # Effect of CPI (higher CPI means less spending money, so lower demand)
    cpi_effect = (102 - market_conditions['cpi']) * 1.0 # Baseline CPI is 102

    # Effect of a promotion/holiday event
    promotion_effect = 20 if market_conditions['is_promotion'] else 0

    # General market randomness/noise
    noise = np.random.normal(0, base_demand * 0.1) # Noise is 10% of base demand

    true_sales = base_demand + price_effect + competitor_effect + cpi_effect + promotion_effect + noise
    
    # Apply the shock and ensure sales are not negative
    return max(0, int(true_sales * shock_multiplier))


def update_market_conditions(market_state):
    """Simulates the market changing from week to week."""
    # CPI has a slight random drift
    market_state['cpi'] += np.random.uniform(-0.1, 0.1)
    
    # Competitor adjusts their price slightly
    market_state['competitor_price'] *= np.random.uniform(0.98, 1.02)
    
    # 15% chance of a promotion event next week
    market_state['is_promotion'] = np.random.rand() < 0.15
    
    return market_state


def run_simulation(num_weeks_to_simulate=10, num_top_products=5, epsilon=0.2, shock_chance=0.1):
    print("--- STARTING ADVANCED DIGITAL TWIN SIMULATION ---")

    # 1. Load initial data and model
    df = pd.read_csv('augmented_processed_data.csv')
    historical_model = joblib.load('historical_model.pkl')
    
    top_products = df['item_description'].value_counts().nlargest(num_top_products).index.tolist()
    print(f"Simulating for top products: {top_products}")

    # Initialize the market state
    market_state = {
        'cpi': 102.0,
        'competitor_price': df['competitor_price'].mean(),
        'is_promotion': False
    }
    
    # NEW: Initialize a list to store the price history
    price_history = []

    # Main simulation loop
    for week in range(1, num_weeks_to_simulate + 1):
        print(f"\n{'='*15} SIMULATING WEEK {week} {'='*15}")
        
        market_state = update_market_conditions(market_state)
        print(f"Market Update: CPI={market_state['cpi']:.2f}, Competitor Price=${market_state['competitor_price']:.2f}, Promotion={market_state['is_promotion']}")
        
        shock_multiplier = 1.0
        if np.random.rand() < shock_chance:
            shock_multiplier = np.random.choice([0.5, 2.0])
            print(f"!!! SHOCK EVENT !!! Sales will be multiplied by {shock_multiplier}")

        df['date'] = pd.to_datetime(df['date'])
        new_weekly_data = []

        # Inner loop for each top product
        for product_name in top_products:
            product_to_simulate = df[df['item_description'] == product_name].sort_values(by='date').iloc[[-1]]
            X_product_base = product_to_simulate.drop(columns=['date', 'supplier', 'item_code', 'item_description', 'warehouse_sales'])
            
            X_product_base['cpi'] = market_state['cpi']
            X_product_base['competitor_price'] = market_state['competitor_price']

            best_price, price_candidates = find_best_price(historical_model, X_product_base)
            final_prescribed_price, strategy = choose_price_strategy(best_price, price_candidates, epsilon)
            
            # NEW: Record the price for this week
            price_history.append({
                'Week': week,
                'Product Name': product_name,
                'Strategy': strategy,
                'Prescribed Price': final_prescribed_price
            })
            
            base_demand = product_to_simulate['warehouse_sales'].values[0]
            base_price = product_to_simulate['retail_sales'].values[0]
            simulated_sales = get_true_market_demand(final_prescribed_price, base_price, base_demand, market_state, shock_multiplier)
            print(f"Product: {product_name[:30]}... | Strategy: {strategy} | Price: ${final_prescribed_price:.2f} | True Sales: {simulated_sales} units")
            
            new_data_row = product_to_simulate.copy()
            new_data_row['warehouse_sales'] = simulated_sales
            new_data_row['retail_sales'] = final_prescribed_price
            new_data_row['cpi'] = market_state['cpi']
            new_data_row['competitor_price'] = market_state['competitor_price']
            new_data_row['date'] = df['date'].max() + pd.Timedelta(weeks=1)
            new_weekly_data.append(new_data_row)

        df = pd.concat([df] + new_weekly_data, ignore_index=True)
        print(f"\nDataset augmented. New size: {df.shape[0]} rows.")
        y = df['warehouse_sales']
        X = df.drop(columns=['date', 'supplier', 'item_code', 'item_description', 'warehouse_sales'])
        historical_model.fit(X, y)
        joblib.dump(historical_model, 'historical_model.pkl')
        print("Model has been retrained on new, more realistic data.")

    print(f"\n{'='*15} SIMULATION COMPLETE {'='*15}")

    # --- NEW: DISPLAY PRICE HISTORY OF SIMULATED PRODUCTS ---
    print(f"\n{'='*15} PRICE EVOLUTION OF TOP PRODUCTS {'='*15}")
    history_df = pd.DataFrame(price_history)
    
    # Pivot the table for easy comparison
    price_pivot_table = history_df.pivot_table(
        index='Product Name',
        columns='Week',
        values='Prescribed Price'
    )
    
    print(price_pivot_table.round(2).to_string())


# Helper functions
def find_best_price(model, X_product_base):
    best_price = 0
    max_profit = -float('inf')
    base_price = X_product_base['retail_sales'].values[0]
    if base_price == 0: base_price = 0.1 
    price_candidates = [base_price * 0.9, base_price, base_price * 1.1]

    for price in price_candidates:
        X_temp = X_product_base.copy()
        X_temp['retail_sales'] = price
        mean_demand, _ = get_probabilistic_forecast(model, X_temp)
        cost = base_price * 0.6
        profit = (price - cost) * mean_demand
        if profit > max_profit:
            max_profit = profit
            best_price = price
    return best_price, price_candidates

def choose_price_strategy(best_price, price_candidates, epsilon):
    if np.random.rand() < epsilon:
        strategy = "Exploration"
        other_prices = [p for p in price_candidates if p != best_price]
        price = np.random.choice(other_prices) if other_prices else best_price
    else:
        strategy = "Exploitation"
        price = best_price
    return price, strategy

def get_probabilistic_forecast(model, data_row):
    predictions = [tree.predict(data_row) for tree in model.estimators_]
    return np.mean(predictions), np.std(predictions)

if __name__ == '__main__':
    run_simulation()