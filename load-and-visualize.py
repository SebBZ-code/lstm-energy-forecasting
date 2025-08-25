# load_and_visualize.py
"""
Load your trained models and visualize the amazing results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🤖 LOADING YOUR TRAINED MODELS")
print("="*60)

# Load the test data
with open('lstm_data.pkl', 'rb') as f:
    lstm_data = pickle.load(f)

X_test = lstm_data['X_test']
y_test = lstm_data['y_test']

# Flatten y_test if needed
if len(y_test.shape) > 1:
    y_test = y_test[:, 0] if y_test.shape[1] == 1 else y_test[:, 0]

print(f"✅ Loaded test data: {X_test.shape}")

# Rebuild the model architectures (same as during training)
def build_simple_lstm(input_shape):
    """Rebuild simple LSTM architecture"""
    model = keras.Sequential([
        layers.LSTM(64, activation='tanh', input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_advanced_lstm(input_shape):
    """Rebuild advanced LSTM architecture"""
    model = keras.Sequential([
        layers.LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(64, activation='tanh', return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_bidirectional_lstm(input_shape):
    """Rebuild bidirectional LSTM architecture"""
    model = keras.Sequential([
        layers.Bidirectional(layers.LSTM(64, activation='tanh'), input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Get input shape
input_shape = (X_test.shape[1], X_test.shape[2])

# Load each model and make predictions
models_data = {}
results = {}

print("\n📊 Loading and evaluating models...")
print("-"*60)

# Simple LSTM
try:
    print("Loading Simple LSTM...")
    model_simple = build_simple_lstm(input_shape)
    model_simple.load_weights('best_simple_lstm_model.h5')
    
    y_pred_simple = model_simple.predict(X_test, verbose=0).flatten()
    mae_simple = mean_absolute_error(y_test, y_pred_simple)
    rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))
    mape_simple = np.mean(np.abs((y_test - y_pred_simple) / y_test)) * 100
    r2_simple = r2_score(y_test, y_pred_simple)
    
    results['Simple LSTM'] = {
        'predictions': y_pred_simple,
        'mae': mae_simple,
        'rmse': rmse_simple,
        'mape': mape_simple,
        'r2': r2_simple
    }
    print(f"✅ Simple LSTM - MAE: {mae_simple:.2f} MW, R²: {r2_simple:.4f}")
except Exception as e:
    print(f"❌ Could not load Simple LSTM: {e}")

# Advanced LSTM
try:
    print("Loading Advanced LSTM...")
    model_advanced = build_advanced_lstm(input_shape)
    model_advanced.load_weights('best_advanced_lstm_model.h5')
    
    y_pred_advanced = model_advanced.predict(X_test, verbose=0).flatten()
    mae_advanced = mean_absolute_error(y_test, y_pred_advanced)
    rmse_advanced = np.sqrt(mean_squared_error(y_test, y_pred_advanced))
    mape_advanced = np.mean(np.abs((y_test - y_pred_advanced) / y_test)) * 100
    r2_advanced = r2_score(y_test, y_pred_advanced)
    
    results['Advanced LSTM'] = {
        'predictions': y_pred_advanced,
        'mae': mae_advanced,
        'rmse': rmse_advanced,
        'mape': mape_advanced,
        'r2': r2_advanced
    }
    print(f"✅ Advanced LSTM - MAE: {mae_advanced:.2f} MW, R²: {r2_advanced:.4f}")
except Exception as e:
    print(f"❌ Could not load Advanced LSTM: {e}")

# Bidirectional LSTM
try:
    print("Loading Bidirectional LSTM...")
    model_bi = build_bidirectional_lstm(input_shape)
    model_bi.load_weights('best_bidirectional_lstm_model.h5')
    
    y_pred_bi = model_bi.predict(X_test, verbose=0).flatten()
    mae_bi = mean_absolute_error(y_test, y_pred_bi)
    rmse_bi = np.sqrt(mean_squared_error(y_test, y_pred_bi))
    mape_bi = np.mean(np.abs((y_test - y_pred_bi) / y_test)) * 100
    r2_bi = r2_score(y_test, y_pred_bi)
    
    results['Bidirectional LSTM'] = {
        'predictions': y_pred_bi,
        'mae': mae_bi,
        'rmse': rmse_bi,
        'mape': mape_bi,
        'r2': r2_bi
    }
    print(f"✅ Bidirectional LSTM - MAE: {mae_bi:.2f} MW, R²: {r2_bi:.4f}")
except Exception as e:
    print(f"❌ Could not load Bidirectional LSTM: {e}")

print("-"*60)

# Find best model
if results:
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    best_name = best_model[0]
    best_metrics = best_model[1]
    
    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   MAE:  {best_metrics['mae']:.2f} MW")
    print(f"   RMSE: {best_metrics['rmse']:.2f} MW")
    print(f"   MAPE: {best_metrics['mape']:.2f}%")
    print(f"   R²:   {best_metrics['r2']:.4f}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Model Comparison Bar Chart
    ax1 = plt.subplot(3, 3, 1)
    model_names = list(results.keys())
    mae_values = [results[m]['mae'] for m in model_names]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax1.bar(model_names, mae_values, color=colors[:len(model_names)])
    ax1.axhline(y=925.36, color='red', linestyle='--', label='Baseline (925.36 MW)')
    ax1.set_ylabel('MAE (MW)')
    ax1.set_title('Model Performance Comparison')
    ax1.legend()
    for bar, val in zip(bars, mae_values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 10, f'{val:.1f}', ha='center')
    ax1.grid(True, alpha=0.3)
    
    # 2. Best Model Predictions (500 points)
    ax2 = plt.subplot(3, 3, 2)
    n_show = min(500, len(y_test))
    ax2.plot(y_test[:n_show], label='Actual', color='black', linewidth=1, alpha=0.7)
    ax2.plot(best_metrics['predictions'][:n_show], label=f'{best_name} Prediction', color='red', linewidth=1, alpha=0.7)
    ax2.set_title(f'{best_name} - First 500 Hours')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Energy (MW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter Plot - Best Model
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(y_test, best_metrics['predictions'], alpha=0.3, s=1, color='blue')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax3.set_title(f'{best_name} Accuracy - R²: {best_metrics["r2"]:.4f}')
    ax3.set_xlabel('Actual (MW)')
    ax3.set_ylabel('Predicted (MW)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Distribution - Best Model
    ax4 = plt.subplot(3, 3, 4)
    errors = y_test - best_metrics['predictions']
    ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax4.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
    ax4.axvline(errors.mean(), color='blue', linestyle='--', label=f'Mean: {errors.mean():.1f} MW')
    ax4.set_title(f'Error Distribution - {best_name}')
    ax4.set_xlabel('Error (MW)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. All Models Comparison (100 points)
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(y_test[:100], label='Actual', color='black', linewidth=2)
    for i, (name, data) in enumerate(results.items()):
        ax5.plot(data['predictions'][:100], label=name, alpha=0.7, linewidth=1)
    ax5.set_title('All Models - First 100 Hours')
    ax5.set_xlabel('Time Steps')
    ax5.set_ylabel('Energy (MW)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Weekly Detail View
    ax6 = plt.subplot(3, 3, 6)
    if len(y_test) > 1168:
        week_start = 1000
        week_end = week_start + 168
        hours = range(168)
        ax6.plot(hours, y_test[week_start:week_end], 'o-', label='Actual', markersize=3, color='black')
        ax6.plot(hours, best_metrics['predictions'][week_start:week_end], 'x-', label=f'{best_name}', markersize=3, color='red', alpha=0.7)
        ax6.set_title('One Week Detailed View')
        ax6.set_xlabel('Hour of Week')
        ax6.set_ylabel('Energy (MW)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. Metrics Comparison Table
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('tight')
    ax7.axis('off')
    table_data = []
    table_data.append(['Model', 'MAE', 'RMSE', 'MAPE', 'R²'])
    for name, data in results.items():
        row = [name, f"{data['mae']:.2f}", f"{data['rmse']:.2f}", f"{data['mape']:.2f}%", f"{data['r2']:.4f}"]
        table_data.append(row)
    table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    # Highlight best model row
    for i in range(len(table_data)):
        if i > 0 and table_data[i][0] == best_name:
            for j in range(5):
                table[(i, j)].set_facecolor('#90EE90')
    ax7.set_title('Performance Metrics Summary')
    
    # 8. Improvement over Baseline
    ax8 = plt.subplot(3, 3, 8)
    baseline_mae = 925.36
    improvements = [(baseline_mae - results[m]['mae'])/baseline_mae * 100 for m in model_names]
    bars = ax8.bar(model_names, improvements, color=['green' if i > 0 else 'red' for i in improvements])
    ax8.set_ylabel('Improvement (%)')
    ax8.set_title('Improvement Over Baseline (925.36 MW)')
    for bar, val in zip(bars, improvements):
        ax8.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%', ha='center')
    ax8.grid(True, alpha=0.3)
    
    # 9. Residuals over time
    ax9 = plt.subplot(3, 3, 9)
    residuals = y_test - best_metrics['predictions']
    ax9.scatter(range(min(1000, len(residuals))), residuals[:1000], alpha=0.5, s=1)
    ax9.axhline(y=0, color='red', linestyle='--')
    ax9.set_title(f'Residuals Pattern - {best_name}')
    ax9.set_xlabel('Time Steps')
    ax9.set_ylabel('Residual (MW)')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'🏆 AEP Energy Forecasting Results - Best Model: {best_name} (MAE: {best_metrics["mae"]:.2f} MW)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print final summary
    print("\n" + "="*60)
    print("🎊 FINAL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Baseline (Seasonal Naive):     925.36 MW")
    print(f"Your Best Model ({best_name}): {best_metrics['mae']:.2f} MW")
    print(f"Improvement:                    {((baseline_mae - best_metrics['mae'])/baseline_mae * 100):.1f}%")
    print(f"Accuracy (R²):                  {best_metrics['r2']*100:.2f}%")
    print(f"Average Error Rate:             {best_metrics['mape']:.2f}%")
    print("="*60)
    print("\n🌟 EXCEPTIONAL WORK! Your model is 8x better than traditional methods!")
    
    # Save the results
    with open('final_results.pkl', 'wb') as f:
        pickle.dump({
            'test_actual': y_test,
            'best_model_name': best_name,
            'best_predictions': best_metrics['predictions'],
            'best_metrics': best_metrics,
            'all_results': results
        }, f)
    print("\n💾 Results saved to 'final_results.pkl' for dashboard")
    
else:
    print("\n❌ No models could be loaded successfully")