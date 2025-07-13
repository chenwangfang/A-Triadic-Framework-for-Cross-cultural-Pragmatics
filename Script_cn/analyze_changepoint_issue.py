#!/usr/bin/env python3
"""
Analyze why Bayesian changepoint detection is showing all changepoints in March 2021
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def analyze_data_flow():
    """Trace the data flow from corpus to changepoint detection"""
    
    print("="*80)
    print("Bayesian Changepoint Detection Data Flow Analysis")
    print("="*80)
    
    # 1. Check the original extracted data
    print("\n1. CHECKING EXTRACTED DATA")
    print("-"*40)
    
    data_path = Path('../output_cn/data')
    
    # Load extracted data
    extracted_df = pd.read_csv(data_path / 'extracted_data.csv', encoding='utf-8-sig')
    extracted_df['date'] = pd.to_datetime(extracted_df['date'])
    
    print(f"Total records in extracted_data.csv: {len(extracted_df)}")
    print(f"Date range: {extracted_df['date'].min()} to {extracted_df['date'].max()}")
    print(f"Unique dates: {extracted_df['date'].nunique()}")
    
    # Check date distribution by year and month
    date_dist = extracted_df.groupby([extracted_df['date'].dt.year, extracted_df['date'].dt.month]).size()
    print("\nDate distribution (Year-Month: Count):")
    for (year, month), count in date_dist.items():
        print(f"  {year}-{month:02d}: {count} records")
    
    # 2. Check the data with metrics
    print("\n2. CHECKING DATA WITH METRICS")
    print("-"*40)
    
    metrics_df = pd.read_csv(data_path / 'data_with_metrics.csv', encoding='utf-8-sig')
    metrics_df['date'] = pd.to_datetime(metrics_df['date'])
    
    print(f"Total records in data_with_metrics.csv: {len(metrics_df)}")
    print(f"Date range: {metrics_df['date'].min()} to {metrics_df['date'].max()}")
    
    # Check constitutive_index distribution
    print(f"\nConstitutive Index stats:")
    print(f"  Mean: {metrics_df['constitutive_index'].mean():.4f}")
    print(f"  Std: {metrics_df['constitutive_index'].std():.4f}")
    print(f"  Min: {metrics_df['constitutive_index'].min():.4f}")
    print(f"  Max: {metrics_df['constitutive_index'].max():.4f}")
    
    # 3. Analyze daily aggregation for changepoint detection
    print("\n3. ANALYZING DAILY AGGREGATION")
    print("-"*40)
    
    # Replicate the aggregation from step5g
    daily_data = metrics_df.groupby('date').agg({
        'dsr_cognitive': 'mean',
        'tl_functional': 'mean',
        'cs_output': 'mean',
        'constitutive_index': 'mean',
        'sensitivity_code': 'mean'
    }).reset_index()
    
    # Calculate constitutive strength
    daily_data['constitutive_strength'] = (
        daily_data['dsr_cognitive'] * 0.3 +
        daily_data['tl_functional'] * 0.3 +
        daily_data['cs_output'] * 0.4
    )
    
    print(f"Daily aggregated records: {len(daily_data)}")
    print(f"Date range: {daily_data['date'].min()} to {daily_data['date'].max()}")
    
    # Check if there's a data gap or anomaly
    date_diffs = daily_data['date'].diff()
    gaps = date_diffs[date_diffs > pd.Timedelta(days=1)]
    
    if len(gaps) > 0:
        print(f"\nFound {len(gaps)} gaps in daily data:")
        for idx in gaps.index:
            print(f"  Gap between {daily_data.loc[idx-1, 'date']} and {daily_data.loc[idx, 'date']}")
    
    # Check constitutive strength over time
    print("\nConstitutive strength by month:")
    monthly_strength = daily_data.groupby(daily_data['date'].dt.to_period('M'))['constitutive_strength'].agg(['mean', 'std', 'count'])
    print(monthly_strength.head(12))  # First year
    
    # 4. Load and analyze changepoint results
    print("\n4. ANALYZING CHANGEPOINT RESULTS")
    print("-"*40)
    
    if (data_path / 'bayesian_changepoint_results.json').exists():
        with open(data_path / 'bayesian_changepoint_results.json', 'r', encoding='utf-8') as f:
            cp_results = json.load(f)
        
        changepoints = cp_results['changepoints']['major_changepoints']
        print(f"Number of changepoints detected: {len(changepoints)}")
        
        if changepoints:
            print("\nChangepoint details:")
            for i, cp in enumerate(changepoints[:10]):  # Show first 10
                print(f"  CP{i+1}: {cp['date'][:10]} (prob={cp['probability']:.4f}, magnitude={cp['magnitude']:.4f})")
            
            # Check if all changepoints are in the same period
            cp_dates = [pd.to_datetime(cp['date']) for cp in changepoints]
            cp_months = set([(d.year, d.month) for d in cp_dates])
            
            if len(cp_months) == 1:
                print(f"\n⚠️  WARNING: All changepoints are in the same month: {cp_months.pop()}")
                print("This suggests a potential issue with:")
                print("  - Data sparsity at the beginning of the time series")
                print("  - Initialization bias in the changepoint detection algorithm")
                print("  - Threshold settings being too sensitive")
    
    # 5. Investigate the issue
    print("\n5. ROOT CAUSE ANALYSIS")
    print("-"*40)
    
    # Check data density at the beginning
    first_month_data = daily_data[daily_data['date'] < '2021-04-01']
    print(f"Data points in March 2021: {len(first_month_data)}")
    
    # Check variance in the first few months
    print("\nVariance analysis by month:")
    for month in pd.date_range('2021-03', '2021-08', freq='M'):
        month_data = daily_data[(daily_data['date'] >= month) & 
                               (daily_data['date'] < month + pd.DateOffset(months=1))]
        if len(month_data) > 0:
            var = month_data['constitutive_strength'].var()
            print(f"  {month.strftime('%Y-%m')}: variance = {var:.6f}, n = {len(month_data)}")
    
    # Check if the algorithm parameters need adjustment
    print("\n6. RECOMMENDATIONS")
    print("-"*40)
    print("1. The changepoint detection threshold (0.01) may be too low")
    print("2. The hazard rate (1/50) assumes changepoints every 50 days on average")
    print("3. Consider using a burn-in period to skip early unstable estimates")
    print("4. The prior parameters may need adjustment based on data characteristics")
    print("5. Consider using a minimum segment length to avoid spurious changepoints")
    
    # Generate a simple visualization of the issue
    print("\n7. GENERATING DIAGNOSTIC PLOT")
    print("-"*40)
    
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Constitutive strength over time
    ax1.plot(daily_data['date'], daily_data['constitutive_strength'], 'b-', alpha=0.7)
    ax1.set_title('Constitutive Strength Over Time')
    ax1.set_ylabel('Constitutive Strength')
    ax1.grid(True, alpha=0.3)
    
    # Mark the period where all changepoints are detected
    ax1.axvspan(pd.to_datetime('2021-03-19'), pd.to_datetime('2021-03-31'), 
                alpha=0.3, color='red', label='Changepoint concentration')
    ax1.legend()
    
    # Plot 2: Daily variance (rolling window)
    rolling_var = daily_data.set_index('date')['constitutive_strength'].rolling('30D').var()
    ax2.plot(rolling_var.index, rolling_var.values, 'g-', alpha=0.7)
    ax2.set_title('30-Day Rolling Variance of Constitutive Strength')
    ax2.set_ylabel('Variance')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../output_cn/figures/changepoint_diagnostic.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Diagnostic plot saved to: ../output_cn/figures/changepoint_diagnostic.jpg")
    
    return daily_data

if __name__ == "__main__":
    daily_data = analyze_data_flow()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)