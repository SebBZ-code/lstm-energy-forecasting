# aep_energy_analysis.py
"""
AEP Energy Consumption Forecasting Project
Complete starter code customized for your dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# ============================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================

def load_aep_data(filepath):
    """Load and prepare AEP energy data"""
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert Datetime to datetime type and set as index
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')
    df.set_index('Datetime', inplace=True)
    
    print("✅ Data loaded successfully!")
    print(f"📊 Dataset range: {df.index[0]} to {df.index[-1]}")
    print(f"📈 Total hours: {len(df):,}")
    
    return df

# ============================================
# STEP 2: FEATURE ENGINEERING
# ============================================

def create_features(df):
    """Create time-based features for better predictions"""
    df = df.copy()
    
    # Time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    
    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for better model understanding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features (previous values)
    for lag in [1, 24, 168]:  # 1 hour ago, 1 day ago, 1 week ago
        df[f'lag_{lag}'] = df['AEP_MW'].shift(lag)
    
    # Rolling statistics
    df['rolling_mean_24h'] = df['AEP_MW'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['AEP_MW'].rolling(window=24).std()
    df['rolling_mean_7d'] = df['AEP_MW'].rolling(window=168).mean()
    
    # Drop rows with NaN from lag/rolling features
    df = df.dropna()
    
    print(f"✅ Created {len(df.columns)-1} features")
    print(f"📊 Dataset after feature engineering: {len(df):,} rows")
    
    return df

# ============================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================

def analyze_patterns(df):
    """Visualize energy consumption patterns"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('AEP Energy Consumption Patterns Analysis', fontsize=16, y=1.02)
    
    # 1. Time series overview
    axes[0, 0].plot(df.index[:8760], df['AEP_MW'][:8760], linewidth=0.5)
    axes[0, 0].set_title('First Year of Data')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Energy (MW)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Hourly pattern
    hourly_avg = df.groupby('hour')['AEP_MW'].mean()
    axes[0, 1].bar(hourly_avg.index, hourly_avg.values, color='skyblue', edgecolor='navy')
    axes[0, 1].set_title('Average Consumption by Hour of Day')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Average MW')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Day of week pattern
    daily_avg = df.groupby('day_of_week')['AEP_MW'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[0, 2].bar(range(7), daily_avg.values, color='lightcoral', edgecolor='darkred')
    axes[0, 2].set_xticks(range(7))
    axes[0, 2].set_xticklabels(days)
    axes[0, 2].set_title('Average Consumption by Day of Week')
    axes[0, 2].set_ylabel('Average MW')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Monthly pattern
    monthly_avg = df.groupby('month')['AEP_MW'].mean()
    axes[1, 0].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8)
    axes[1, 0].set_title('Average Consumption by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average MW')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Distribution
    axes[1, 1].hist(df['AEP_MW'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].axvline(df['AEP_MW'].mean(), color='red', linestyle='--', label=f'Mean: {df["AEP_MW"].mean():.0f} MW')
    axes[1, 1].set_title('Energy Consumption Distribution')
    axes[1, 1].set_xlabel('Energy (MW)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Yearly trend
    yearly_avg = df.groupby('year')['AEP_MW'].mean()
    axes[1, 2].plot(yearly_avg.index, yearly_avg.values, marker='s', linewidth=2, markersize=8, color='purple')
    axes[1, 2].set_title('Yearly Average Consumption Trend')
    axes[1, 2].set_xlabel('Year')
    axes[1, 2].set_ylabel('Average MW')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Weekend vs Weekday
    weekend_data = df[df['is_weekend'] == 1]['AEP_MW']
    weekday_data = df[df['is_weekend'] == 0]['AEP_MW']
    axes[2, 0].boxplot([weekday_data, weekend_data], labels=['Weekday', 'Weekend'])
    axes[2, 0].set_title('Weekday vs Weekend Consumption')
    axes[2, 0].set_ylabel('Energy (MW)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Seasonal pattern (using quarters as proxy)
    seasonal_avg = df.groupby('quarter')['AEP_MW'].mean()
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    axes[2, 1].bar(range(4), seasonal_avg.values, color=['lightblue', 'lightgreen', 'yellow', 'orange'])
    axes[2, 1].set_xticks(range(4))
    axes[2, 1].set_xticklabels(seasons)
    axes[2, 1].set_title('Seasonal Consumption Pattern')
    axes[2, 1].set_ylabel('Average MW')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Autocorrelation
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(df['AEP_MW'][:1000], ax=axes[2, 2])
    axes[2, 2].set_title('Autocorrelation (First 1000 hours)')
    axes[2, 2].set_xlabel('Lag')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print("\n📊 KEY INSIGHTS FROM YOUR DATA:")
    print(f"• Peak hour: {hourly_avg.idxmax()}:00 (avg {hourly_avg.max():.0f} MW)")
    print(f"• Lowest hour: {hourly_avg.idxmin()}:00 (avg {hourly_avg.min():.0f} MW)")
    print(f"• Peak day: {days[daily_avg.idxmax()]} (avg {daily_avg.max():.0f} MW)")
    print(f"• Peak month: Month {monthly_avg.idxmax()} (avg {monthly_avg.max():.0f} MW)")
    print(f"• Weekend avg: {weekend_data.mean():.0f} MW")
    print(f"• Weekday avg: {weekday_data.mean():.0f} MW")

# ============================================
# STEP 4: PREPARE DATA FOR LSTM
# ============================================

def prepare_sequences(df, target_col='AEP_MW', lookback=24, forecast_horizon=24):
    """
    Prepare sequences for LSTM training
    
    Args:
        df: DataFrame with features
        target_col: Column to predict
        lookback: Hours to look back
        forecast_horizon: Hours to predict ahead
    """
    # Select features for model
    feature_cols = ['AEP_MW', 'hour_sin', 'hour_cos', 'day_of_week', 
                    'month_sin', 'month_cos', 'is_weekend',
                    'lag_24', 'rolling_mean_24h']
    
    # Ensure all features exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Prepare data
    data = df[feature_cols].values
    target_data = df[target_col].values
    
    # Scale features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(data_scaled) - forecast_horizon):
        X.append(data_scaled[i-lookback:i])
        y.append(target_data[i:i+forecast_horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    # Train/validation/test split (70/15/15)
    n_train = int(0.7 * len(X))
    n_val = int(0.85 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_val], y[n_train:n_val]
    X_test, y_test = X[n_val:], y[n_val:]
    
    print(f"\n📊 LSTM Data Shapes:")
    print(f"X_train: {X_train.shape} -> (samples, lookback, features)")
    print(f"y_train: {y_train.shape} -> (samples, forecast_horizon)")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols
    }

# ============================================
# STEP 5: BASELINE MODELS
# ============================================

def create_baseline_models(df, test_size=0.15):
    """Create simple baseline models for comparison"""
    
    # Split data
    split_point = int(len(df) * (1 - test_size))
    train = df['AEP_MW'][:split_point]
    test = df['AEP_MW'][split_point:]
    
    results = {}
    
    # 1. Naive (last value)
    naive_pred = np.roll(test.values, 1)
    naive_pred[0] = train.values[-1]
    results['Naive'] = {
        'predictions': naive_pred,
        'mae': mean_absolute_error(test.values, naive_pred),
        'rmse': np.sqrt(mean_squared_error(test.values, naive_pred))
    }
    
    # 2. Seasonal Naive (same hour yesterday)
    seasonal_pred = np.roll(test.values, 24)
    seasonal_pred[:24] = train.values[-24:]
    results['Seasonal Naive (24h)'] = {
        'predictions': seasonal_pred,
        'mae': mean_absolute_error(test.values, seasonal_pred),
        'rmse': np.sqrt(mean_squared_error(test.values, seasonal_pred))
    }
    
    # 3. Moving Average
    window = 168  # 1 week
    ma_pred = pd.Series(df['AEP_MW'].values).rolling(window=window).mean()
    ma_pred = ma_pred[split_point:].values
    results['Moving Avg (7 days)'] = {
        'predictions': ma_pred,
        'mae': mean_absolute_error(test.values, ma_pred),
        'rmse': np.sqrt(mean_squared_error(test.values, ma_pred))
    }
    
    # Print results
    print("\n📊 BASELINE MODEL RESULTS:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:20} | MAE: {metrics['mae']:7.2f} | RMSE: {metrics['rmse']:7.2f}")
    print("-" * 50)
    print("🎯 Your LSTM should beat these scores!\n")
    
    return results, test

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    
    print("="*60)
    print("⚡ AEP ENERGY FORECASTING PROJECT")
    print("="*60)
    
    # Load data - UPDATE THIS PATH
    filepath = r'C:\Users\sebas\Documents\CODE\SebBZ-codesbz-ML-AI\sbz-ML-AI\LSTM Project\AEP_hourly.csv'
    df = load_aep_data(filepath)
    
    print("\n" + "="*60)
    print("📊 FEATURE ENGINEERING")
    print("="*60)
    df_features = create_features(df)
    
    print("\n" + "="*60)
    print("📈 EXPLORATORY DATA ANALYSIS")
    print("="*60)
    analyze_patterns(df_features)
    
    print("\n" + "="*60)
    print("🎯 BASELINE MODELS")
    print("="*60)
    baseline_results, test_data = create_baseline_models(df_features)
    
    print("\n" + "="*60)
    print("🤖 PREPARING LSTM DATA")
    print("="*60)
    lstm_data = prepare_sequences(df_features, lookback=24, forecast_horizon=1)
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\n🚀 NEXT STEPS:")
    print("1. Build LSTM model using the prepared sequences")
    print("2. Train model and beat baseline MAE:", 
          f"{baseline_results['Seasonal Naive (24h)']['mae']:.2f}")
    print("3. Create visualizations in Streamlit")
    print("4. Document your findings")
    
    # Save prepared data for later use
    import pickle
    with open('lstm_data.pkl', 'wb') as f:
        pickle.dump(lstm_data, f)
    print("\n💾 Data saved to 'lstm_data.pkl' for model training")
    
    return df_features, lstm_data, baseline_results

if __name__ == "__main__":
    df, lstm_data, baselines = main()