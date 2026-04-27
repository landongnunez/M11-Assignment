# Module 11 Assignment: Data Visualization with Matplotlib
# SunCoast Retail Visual Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- DATA CREATION (DO NOT MODIFY) -----
np.random.seed(42)
quarters = pd.date_range(start='2022-01-01', periods=8, freq='Q')
quarter_labels = ['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022', 
                  'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023']
locations = ['Tampa', 'Miami', 'Orlando', 'Jacksonville']
categories = ['Electronics', 'Clothing', 'Home Goods', 'Sporting Goods', 'Beauty']

quarterly_data = []
for quarter_idx, quarter in enumerate(quarters):
    for location in locations:
        for category in categories:
            base_sales = np.random.normal(loc=100000, scale=20000)
            seasonal_factor = 1.3 if quarter.quarter == 4 else (0.8 if quarter.quarter == 1 else 1.0)
            location_factor = {'Tampa': 1.0, 'Miami': 1.2, 'Orlando': 0.9, 'Jacksonville': 0.8}[location]
            category_factor = {'Electronics': 1.5, 'Clothing': 1.0, 'Home Goods': 0.8, 'Sporting Goods': 0.7, 'Beauty': 0.9}[category]
            growth_factor = (1 + 0.05/4) ** quarter_idx
            sales = base_sales * seasonal_factor * location_factor * category_factor * growth_factor
            sales = sales * np.random.normal(loc=1.0, scale=0.1)
            ad_spend = (sales ** 0.7) * 0.05 * np.random.normal(loc=1.0, scale=0.2)
            quarterly_data.append({
                'Quarter': quarter, 'QuarterLabel': quarter_labels[quarter_idx],
                'Location': location, 'Category': category, 'Sales': round(sales, 2),
                'AdSpend': round(ad_spend, 2), 'Year': quarter.year
            })

customer_data = []
age_params = {'Tampa': (45, 15), 'Miami': (35, 12), 'Orlando': (38, 14), 'Jacksonville': (42, 13)}
for location in locations:
    mean_age, std_age = age_params[location]
    customer_count = int(2000 * {'Tampa': 0.3, 'Miami': 0.35, 'Orlando': 0.2, 'Jacksonville': 0.15}[location])
    ages = np.clip(np.random.normal(loc=mean_age, scale=std_age, size=customer_count), 18, 80).astype(int)
    for age in ages:
        p = [0.3, 0.3, 0.1, 0.2, 0.1] if age < 30 else ([0.25, 0.2, 0.25, 0.15, 0.15] if age < 50 else [0.15, 0.1, 0.35, 0.1, 0.3])
        cat = np.random.choice(categories, p=p)
        tier = np.random.choice(['Budget', 'Mid-range', 'Premium'], p=[0.3, 0.5, 0.2])
        amt = np.random.gamma(shape=5, scale=20) * {'Budget': 0.7, 'Mid-range': 1.0, 'Premium': 1.8}[tier]
        customer_data.append({'Location': location, 'Age': age, 'Category': cat, 'PurchaseAmount': round(amt, 2), 'PriceTier': tier})

sales_df = pd.DataFrame(quarterly_data)
customer_df = pd.DataFrame(customer_data)
sales_df['SalesPerDollarSpent'] = sales_df['Sales'] / sales_df['AdSpend']

# ----- VISUALIZATION FUNCTIONS -----

def plot_quarterly_sales_trend():
    fig, ax = plt.subplots(figsize=(10, 6))
    trend = sales_df.groupby('QuarterLabel')['Sales'].sum().reindex(quarter_labels)
    ax.plot(trend.index, trend.values, marker='o', linestyle='-', color='teal', linewidth=2)
    ax.set_title('Overall Quarterly Sales Trend (2022-2023)', fontsize=14)
    ax.set_ylabel('Total Sales ($)')
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def plot_location_sales_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    for loc in locations:
        loc_data = sales_df[sales_df['Location'] == loc].groupby('QuarterLabel')['Sales'].sum().reindex(quarter_labels)
        ax.plot(loc_data.index, loc_data.values, marker='s', label=loc)
    ax.set_title('Quarterly Sales Comparison by Location', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_category_performance_by_location():
    fig, ax = plt.subplots(figsize=(10, 6))
    latest_q = sales_df[sales_df['QuarterLabel'] == 'Q4 2023']
    pivot_data = latest_q.pivot_table(index='Category', columns='Location', values='Sales', aggfunc='sum')
    pivot_data.plot(kind='bar', ax=ax)
    ax.set_title('Category Performance by Location (Q4 2023)')
    ax.set_ylabel('Sales ($)')
    plt.xticks(rotation=45)
    return fig

def plot_sales_composition_by_location():
    fig, ax = plt.subplots(figsize=(10, 6))
    composition = sales_df.pivot_table(index='Location', columns='Category', values='Sales', aggfunc='sum')
    composition_pct = composition.div(composition.sum(axis=1), axis=0) * 100
    composition_pct.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title('Sales Composition % by Location')
    ax.set_ylabel('Percentage of Total Sales')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig

def plot_ad_spend_vs_sales():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sales_df['AdSpend'], sales_df['Sales'], alpha=0.6, color='coral')
    m, b = np.polyfit(sales_df['AdSpend'], sales_df['Sales'], 1)
    ax.plot(sales_df['AdSpend'], m*sales_df['AdSpend'] + b, color='darkred', linestyle='--')
    ax.set_title('Advertising Spend vs. Total Sales')
    ax.set_xlabel('Ad Spend ($)')
    ax.set_ylabel('Sales ($)')
    return fig

def plot_ad_efficiency_over_time():
    fig, ax = plt.subplots(figsize=(10, 6))
    efficiency = sales_df.groupby('QuarterLabel')['SalesPerDollarSpent'].mean().reindex(quarter_labels)
    ax.plot(efficiency.index, efficiency.values, color='green', marker='^')
    ax.set_title('Advertising Efficiency Over Time')
    ax.set_ylabel('Sales per Ad Dollar ($)')
    return fig

def plot_customer_age_distribution():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    # Overall
    axes[0].hist(customer_df['Age'], bins=15, color='skyblue', edgecolor='black')
    axes[0].axvline(customer_df['Age'].mean(), color='red', label='Mean')
    axes[0].set_title('Overall Age Distribution')
    # Per Location
    for i, loc in enumerate(locations):
        data = customer_df[customer_df['Location'] == loc]['Age']
        axes[i+1].hist(data, bins=15, color='orange', alpha=0.7)
        axes[i+1].set_title(f'Age Dist: {loc}')
    plt.tight_layout()
    return fig

def plot_purchase_by_age_group():
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = [18, 30, 45, 60, 80]
    labels = ['18-30', '31-45', '46-60', '61+']
    customer_df['AgeGroup'] = pd.cut(customer_df['Age'], bins=bins, labels=labels)
    customer_df.boxplot(column='PurchaseAmount', by='AgeGroup', ax=ax)
    ax.set_title('Purchase Amount by Age Group')
    plt.suptitle('') # Remove default pandas title
    return fig

def plot_purchase_amount_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(customer_df['PurchaseAmount'], bins=30, color='purple', alpha=0.7)
    ax.set_title('Distribution of Individual Purchase Amounts')
    ax.set_xlabel('Amount ($)')
    return fig

def plot_sales_by_price_tier():
    fig, ax = plt.subplots()
    tier_sales = customer_df.groupby('PriceTier')['PurchaseAmount'].sum()
    explode = [0.1 if i == tier_sales.idxmax() else 0 for i in tier_sales.index]
    ax.pie(tier_sales, labels=tier_sales.index, autopct='%1.1f%%', explode=explode, startangle=140)
    ax.set_title('Sales Breakdown by Price Tier')
    return fig

def plot_category_market_share():
    fig, ax = plt.subplots()
    cat_sales = sales_df.groupby('Category')['Sales'].sum()
    explode = [0.1 if i == cat_sales.idxmax() else 0 for i in cat_sales.index]
    ax.pie(cat_sales, labels=cat_sales.index, autopct='%1.1f%%', explode=explode)
    ax.set_title('Market Share by Category')
    return fig

def plot_location_sales_distribution():
    fig, ax = plt.subplots()
    loc_sales = sales_df.groupby('Location')['Sales'].sum()
    ax.pie(loc_sales, labels=loc_sales.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Total Sales Distribution by Location')
    return fig

def create_business_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Trend
    trend = sales_df.groupby('QuarterLabel')['Sales'].sum().reindex(quarter_labels)
    axes[0,0].plot(trend.index, trend.values, marker='o')
    axes[0,0].set_title('Quarterly Revenue Trend')
    
    # 2. Location Share
    loc_sales = sales_df.groupby('Location')['Sales'].sum()
    axes[0,1].pie(loc_sales, labels=loc_sales.index, autopct='%1.1f%%')
    axes[0,1].set_title('Revenue Share by City')
    
    # 3. Ad Efficiency
    eff = sales_df.groupby('QuarterLabel')['SalesPerDollarSpent'].mean().reindex(quarter_labels)
    axes[1,0].bar(eff.index, eff.values, color='lightgreen')
    axes[1,0].set_title('Ad Efficiency (ROI)')
    
    # 4. Age Dist
    axes[1,1].hist(customer_df['Age'], bins=20, color='gray')
    axes[1,1].set_title('Customer Age Profile')
    
    fig.suptitle('SunCoast Retail Executive Dashboard', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def main():
    print("\n" + "=" * 60)
    print("SUNCOAST RETAIL VISUAL ANALYSIS RESULTS")
    print("=" * 60)

    # Call all visualization functions
    fig1 = plot_quarterly_sales_trend()
    fig2 = plot_location_sales_comparison()
    fig3 = plot_category_performance_by_location()
    fig4 = plot_sales_composition_by_location()
    fig5 = plot_ad_spend_vs_sales()
    fig6 = plot_ad_efficiency_over_time()
    fig7 = plot_customer_age_distribution()
    fig8 = plot_purchase_by_age_group()
    fig9 = plot_purchase_amount_distribution()
    fig10 = plot_sales_by_price_tier()
    fig11 = plot_category_market_share()
    fig12 = plot_location_sales_distribution()
    fig13 = create_business_dashboard()

    print("\nKEY BUSINESS INSIGHTS:")
    print("1. Seasonal Performance: Significant sales spikes occur in Q4 annually across all locations.")
    print("2. Regional Leaders: Miami consistently outperforms other locations, holding the largest market share.")
    print("3. Ad Efficiency: There is a strong linear correlation between ad spend and sales, though ROI fluctuates quarterly.")
    print("4. Demographics: Tampa serves an older demographic (mean ~45) compared to Miami (~35), influencing category preferences.")
    print("5. Pricing: Mid-range products dominate the sales volume, contributing to roughly 50% of revenue.")

    plt.show()

if __name__ == "__main__":
    main()