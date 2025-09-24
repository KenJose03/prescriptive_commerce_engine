import pandas as pd
import numpy as np
import joblib
import warnings

# Suppress the UserWarning from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)


def get_true_market_demand(prescribed_price, base_price, base_demand, market_conditions, shock_multiplier=1.0):
    """This is our 'hidden' ground truth model."""
    price_effect = (base_price - prescribed_price) * 2.5
    competitor_effect = (market_conditions['competitor_price'] - prescribed_price) * 0.5
    cpi_effect = (102 - market_conditions['cpi']) * 1.0
    promotion_effect = 20 if market_conditions['is_promotion'] else 0
    noise = np.random.normal(0, base_demand * 0.1)
    true_sales = base_demand + price_effect + competitor_effect + cpi_effect + promotion_effect + noise
    return max(0, int(true_sales * shock_multiplier))


def update_market_conditions(market_state):
    """Simulates the market changing from week to week."""
    market_state['cpi'] += np.random.uniform(-0.1, 0.1)
    market_state['competitor_price'] *= np.random.uniform(0.98, 1.02)
    market_state['is_promotion'] = np.random.rand() < 0.15
    return market_state


def run_simulation(num_weeks_to_simulate=10, epsilon=0.2, shock_chance=0.1):
    print("--- STARTING ADVANCED DIGITAL TWIN SIMULATION ---")

    # 1. Load initial data and model
    df = pd.read_csv('augmented_processed_data.csv')
    historical_model = joblib.load('historical_model.pkl')
    
    # Stratified Sampling for 10 Dynamic Products
    print("Selecting a dynamic, stratified sample of 10 products...")
    product_counts = df['item_description'].value_counts()
    top_tier_cutoff = product_counts.quantile(0.8); mid_tier_cutoff = product_counts.quantile(0.4)
    top_sellers = product_counts[product_counts >= top_tier_cutoff].index.tolist()
    mid_sellers = product_counts[(product_counts >= mid_tier_cutoff) & (product_counts < top_tier_cutoff)].index.tolist()
    low_sellers = product_counts[product_counts < mid_tier_cutoff].index.tolist()
    num_top = min(len(top_sellers), 5); num_mid = min(len(mid_sellers), 3); num_low = min(len(low_sellers), 2)
    selected_top = np.random.choice(top_sellers, num_top, replace=False).tolist()
    selected_mid = np.random.choice(mid_sellers, num_mid, replace=False).tolist()
    selected_low = np.random.choice(low_sellers, num_low, replace=False).tolist()
    simulated_products = selected_top + selected_mid + selected_low
    print(f"Simulating for a diverse set of {len(simulated_products)} products.")

    # Calculate initial average sales for these products before the simulation
    initial_avg_sales = df[df['item_description'].isin(simulated_products)].groupby('item_description')['warehouse_sales'].mean().to_dict()

    # Initialize the market state and history tracker
    market_state = { 'cpi': 102.0, 'competitor_price': df['competitor_price'].mean(), 'is_promotion': False }
    simulation_history = []

    # Main simulation loop
    for week in range(1, num_weeks_to_simulate + 1):
        print(f"\n{'='*15} SIMULATING WEEK {week} {'='*15}")
        market_state = update_market_conditions(market_state)
        print(f"Market Update: CPI={market_state['cpi']:.2f}, Competitor Price=${market_state['competitor_price']:.2f}, Promotion={market_state['is_promotion']}")
        
        shock_multiplier = 1.0
        if np.random.rand() < shock_chance:
            shock_multiplier = np.random.choice([0.5, 2.0]); print(f"!!! SHOCK EVENT !!! Sales will be multiplied by {shock_multiplier}")

        df['date'] = pd.to_datetime(df['date'])
        new_weekly_data = []

        # Inner loop for each selected product
        for product_name in simulated_products:
            product_to_simulate = df[df['item_description'] == product_name].sort_values(by='date').iloc[[-1]]
            X_product_base = product_to_simulate.drop(columns=['date', 'supplier', 'item_code', 'item_description', 'warehouse_sales'])
            X_product_base['cpi'] = market_state['cpi']; X_product_base['competitor_price'] = market_state['competitor_price']

            best_price, price_candidates = find_best_price(historical_model, X_product_base)
            final_prescribed_price, strategy = choose_price_strategy(best_price, price_candidates, epsilon)
            
            base_demand = product_to_simulate['warehouse_sales'].values[0]
            base_price = product_to_simulate['retail_sales'].values[0]
            simulated_sales = get_true_market_demand(final_prescribed_price, base_price, base_demand, market_state, shock_multiplier)
            print(f"Product: {product_name[:30]}... | Strategy: {strategy} | Price: ${final_prescribed_price:.2f} | True Sales: {simulated_sales} units")
            
            simulation_history.append({'Week': week, 'Product Name': product_name, 'Strategy': strategy, 'Prescribed Price': final_prescribed_price, 'Simulated Sales': simulated_sales})
            
            new_data_row = product_to_simulate.copy()
            new_data_row['warehouse_sales'] = simulated_sales; new_data_row['retail_sales'] = final_prescribed_price
            new_data_row['cpi'] = market_state['cpi']; new_data_row['competitor_price'] = market_state['competitor_price']
            new_data_row['date'] = df['date'].max() + pd.Timedelta(weeks=1)
            new_weekly_data.append(new_data_row)

        df = pd.concat([df] + new_weekly_data, ignore_index=True)
        print(f"\nDataset augmented. New size: {df.shape[0]} rows.")
        y = df['warehouse_sales']
        X = df.drop(columns=['date', 'supplier', 'item_code', 'item_description', 'warehouse_sales'])
        historical_model.fit(X, y); joblib.dump(historical_model, 'historical_model.pkl')
        print("Model has been retrained on new, more realistic data.")

    print(f"\n{'='*15} SIMULATION COMPLETE {'='*15}")
    
    # --- DISPLAY PRICE EVOLUTION ---
    print(f"\n{'='*15} PRICE EVOLUTION OF SIMULATED PRODUCTS {'='*15}")
    history_df = pd.DataFrame(simulation_history)
    price_pivot_table = history_df.pivot_table(index='Product Name', columns='Week', values='Prescribed Price')
    print(price_pivot_table.round(2).to_string())

    # --- DISPLAY SALES PERFORMANCE SUMMARY ---
    print(f"\n{'='*15} SALES PERFORMANCE SUMMARY {'='*15}")
    final_avg_sales = history_df.groupby('Product Name')['Simulated Sales'].mean()

    summary_data = []
    for product in simulated_products:
        initial = initial_avg_sales.get(product, 0)
        final = final_avg_sales.get(product, 0)
        
        # --- UPDATED: Handle division by zero and replace inf ---
        if initial > 0:
            increase_pct = ((final - initial) / initial) * 100
        else:
            # If initial sales were 0, any new sales are a new product sale
            increase_pct = "New Sale" if final > 0 else 0.0

        summary_data.append({
            'Product Name': product,
            'Initial Avg Sales': initial,
            'Simulated Avg Sales': final,
            'Increase (%)': increase_pct
        })

    summary_df = pd.DataFrame(summary_data)
    
    # Custom formatting to handle the new string value
    def format_increase(val):
        if isinstance(val, str):
            return val
        return f"{val:.2f}"

    summary_df['Initial Avg Sales'] = summary_df['Initial Avg Sales'].round(2)
    summary_df['Simulated Avg Sales'] = summary_df['Simulated Avg Sales'].round(2)
    summary_df['Increase (%)'] = summary_df['Increase (%)'].apply(format_increase)

    print(summary_df.to_string())


# Helper functions
def find_best_price(model, X_product_base):
    best_price = 0; max_profit = -float('inf'); base_price = X_product_base['retail_sales'].values[0]
    if base_price == 0: base_price = 0.1 
    price_candidates = [base_price * 0.9, base_price, base_price * 1.1]
    for price in price_candidates:
        X_temp = X_product_base.copy()
        X_temp['retail_sales'] = price
        mean_demand, _ = get_probabilistic_forecast(model, X_temp)
        cost = base_price * 0.6
        profit = (price - cost) * mean_demand
        if profit > max_profit: max_profit = profit; best_price = price
    return best_price, price_candidates

def choose_price_strategy(best_price, price_candidates, epsilon):
    if np.random.rand() < epsilon:
        strategy = "Exploration"; price = np.random.choice([p for p in price_candidates if p != best_price] or [best_price])
    else:
        strategy = "Exploitation"; price = best_price
    return price, strategy

def get_probabilistic_forecast(model, data_row):
    predictions = [tree.predict(data_row) for tree in model.estimators_]
    return np.mean(predictions), np.std(predictions)

if __name__ == '__main__':
    run_simulation()