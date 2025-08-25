# aep_lstm_model.py
"""
LSTM Model for AEP Energy Forecasting
This builds and trains the actual neural network
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("🤖 AEP ENERGY LSTM MODEL TRAINING")
print("="*60)

# ============================================
# STEP 1: LOAD PREPARED DATA
# ============================================

print("\n📂 Loading prepared data...")
try:
    with open('lstm_data.pkl', 'rb') as f:
        lstm_data = pickle.load(f)
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ Please run aep_energy_analysis.py first to prepare the data!")
    exit()

# Extract data
X_train = lstm_data['X_train']
y_train = lstm_data['y_train']
X_val = lstm_data['X_val']
y_val = lstm_data['y_val']
X_test = lstm_data['X_test']
y_test = lstm_data['y_test']

# If y is 2D (multiple forecast steps), flatten it to 1D
if len(y_train.shape) > 1 and y_train.shape[1] == 1:
    y_train = y_train.flatten()
    y_val = y_val.flatten()
    y_test = y_test.flatten()
elif len(y_train.shape) > 1:
    # Take only first prediction if multiple horizons
    y_train = y_train[:, 0]
    y_val = y_val[:, 0]
    y_test = y_test[:, 0]

print(f"\n📊 Data shapes:")
print(f"Training:   X: {X_train.shape}, y: {y_train.shape}")
print(f"Validation: X: {X_val.shape}, y: {y_val.shape}")
print(f"Test:       X: {X_test.shape}, y: {y_test.shape}")

# ============================================
# STEP 2: BUILD LSTM MODELS
# ============================================

def build_simple_lstm(input_shape, learning_rate=0.001):
    """Simple LSTM - start here"""
    model = keras.Sequential([
        # LSTM layer
        layers.LSTM(64, activation='tanh', input_shape=input_shape),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_advanced_lstm(input_shape, learning_rate=0.001):
    """More complex LSTM - try after simple works"""
    model = keras.Sequential([
        # First LSTM layer
        layers.LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        
        # Second LSTM layer  
        layers.LSTM(64, activation='tanh', return_sequences=False),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_bidirectional_lstm(input_shape, learning_rate=0.001):
    """Bidirectional LSTM - captures patterns from both directions"""
    model = keras.Sequential([
        # Bidirectional LSTM
        layers.Bidirectional(layers.LSTM(64, activation='tanh'), input_shape=input_shape),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ============================================
# STEP 3: TRAINING FUNCTION
# ============================================

def train_model(model, X_train, y_train, X_val, y_val, model_name="LSTM", patience=15):
    """Train the model with callbacks"""
    
    print(f"\n🚀 Training {model_name} Model...")
    print(f"Parameters: {model.count_params():,}")
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        f'best_{model_name.lower()}_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    return history

# ============================================
# STEP 4: EVALUATION FUNCTIONS
# ============================================

def evaluate_model(model, X_test, y_test, model_name="LSTM"):
    """Evaluate model performance"""
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 {model_name} Model Performance:")
    print(f"MAE:  {mae:.2f} MW")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")
    
    return y_pred, {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}

def plot_results(y_test, predictions_dict, history_dict):
    """Plot all results"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training history comparison
    ax1 = plt.subplot(3, 3, 1)
    for name, history in history_dict.items():
        ax1.plot(history.history['loss'], label=f'{name} Train', alpha=0.7)
        ax1.plot(history.history['val_loss'], label=f'{name} Val', alpha=0.7, linestyle='--')
    ax1.set_title('Model Training Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE comparison
    ax2 = plt.subplot(3, 3, 2)
    for name, history in history_dict.items():
        ax2.plot(history.history['mae'], label=f'{name} Train', alpha=0.7)
        ax2.plot(history.history['val_mae'], label=f'{name} Val', alpha=0.7, linestyle='--')
    ax2.set_title('MAE During Training')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Predictions comparison (first 500 points)
    ax3 = plt.subplot(3, 3, 3)
    n_points = min(500, len(y_test))
    ax3.plot(y_test[:n_points], label='Actual', color='black', alpha=0.7, linewidth=1)
    colors = ['blue', 'red', 'green']
    for i, (name, preds) in enumerate(predictions_dict.items()):
        ax3.plot(preds[:n_points], label=name, alpha=0.6, linewidth=1, color=colors[i % 3])
    ax3.set_title('Predictions vs Actual (First 500 hours)')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Energy (MW)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Scatter plots for each model
    for idx, (name, preds) in enumerate(predictions_dict.items()):
        ax = plt.subplot(3, 3, 4 + idx)
        ax.scatter(y_test, preds, alpha=0.3, s=1)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_title(f'{name} Accuracy')
        ax.set_xlabel('Actual (MW)')
        ax.set_ylabel('Predicted (MW)')
        ax.grid(True, alpha=0.3)
    
    # 7. Error distribution
    ax7 = plt.subplot(3, 3, 7)
    for name, preds in predictions_dict.items():
        errors = y_test - preds
        ax7.hist(errors, bins=50, alpha=0.5, label=name, edgecolor='black')
    ax7.set_title('Prediction Error Distribution')
    ax7.set_xlabel('Error (MW)')
    ax7.set_ylabel('Frequency')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Zoomed prediction (24 hours)
    ax8 = plt.subplot(3, 3, 8)
    zoom_start = 1000
    zoom_end = zoom_start + 168  # 1 week
    ax8.plot(y_test[zoom_start:zoom_end], label='Actual', color='black', marker='o', markersize=3)
    for name, preds in predictions_dict.items():
        ax8.plot(preds[zoom_start:zoom_end], label=name, alpha=0.7, marker='x', markersize=3)
    ax8.set_title('One Week Detailed View')
    ax8.set_xlabel('Hours')
    ax8.set_ylabel('Energy (MW)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Metrics comparison bar chart
    ax9 = plt.subplot(3, 3, 9)
    metrics_names = list(predictions_dict.keys())
    mae_values = [metrics[name]['mae'] for name in metrics_names]
    rmse_values = [metrics[name]['rmse'] for name in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax9.bar(x - width/2, mae_values, width, label='MAE', color='skyblue')
    ax9.bar(x + width/2, rmse_values, width, label='RMSE', color='lightcoral')
    ax9.set_xlabel('Model')
    ax9.set_ylabel('Error (MW)')
    ax9.set_title('Model Performance Comparison')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics_names)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('AEP Energy Forecasting - Model Results', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# ============================================
# STEP 5: MAIN EXECUTION
# ============================================

# Build models
print("\n🔨 Building models...")
input_shape = (X_train.shape[1], X_train.shape[2])

# Model 1: Simple LSTM
model_simple = build_simple_lstm(input_shape)
print("\n📊 Simple LSTM Architecture:")
model_simple.summary()

# Model 2: Advanced LSTM
model_advanced = build_advanced_lstm(input_shape)

# Model 3: Bidirectional LSTM
model_bidirectional = build_bidirectional_lstm(input_shape)

# Train models
print("\n" + "="*60)
print("🚀 TRAINING PHASE")
print("="*60)

# Train Simple LSTM
history_simple = train_model(model_simple, X_train, y_train, X_val, y_val, "Simple_LSTM", patience=10)

# Train Advanced LSTM
history_advanced = train_model(model_advanced, X_train, y_train, X_val, y_val, "Advanced_LSTM", patience=10)

# Train Bidirectional LSTM
history_bidirectional = train_model(model_bidirectional, X_train, y_train, X_val, y_val, "Bidirectional_LSTM", patience=10)

# Evaluate models
print("\n" + "="*60)
print("📊 EVALUATION PHASE")
print("="*60)

predictions = {}
metrics = {}

# Evaluate each model
pred_simple, metrics['Simple'] = evaluate_model(model_simple, X_test, y_test, "Simple LSTM")
predictions['Simple LSTM'] = pred_simple

pred_advanced, metrics['Advanced'] = evaluate_model(model_advanced, X_test, y_test, "Advanced LSTM")
predictions['Advanced LSTM'] = pred_advanced

pred_bidirectional, metrics['Bidirectional'] = evaluate_model(model_bidirectional, X_test, y_test, "Bidirectional LSTM")
predictions['Bidirectional LSTM'] = pred_bidirectional

# Find best model
best_model = min(metrics.items(), key=lambda x: x[1]['mae'])
print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model[0]} with MAE: {best_model[1]['mae']:.2f} MW")
print("="*60)

# Plot all results
history_dict = {
    'Simple': history_simple,
    'Advanced': history_advanced,
    'Bidirectional': history_bidirectional
}

plot_results(y_test, predictions, history_dict)

# Save best model
if best_model[0] == 'Simple':
    best_model_obj = model_simple
elif best_model[0] == 'Advanced':
    best_model_obj = model_advanced
else:
    best_model_obj = model_bidirectional

best_model_obj.save('best_aep_lstm_model.h5')
print(f"\n💾 Best model saved as 'best_aep_lstm_model.h5'")

# Create predictions for dashboard
print("\n📊 Creating forecast for next 24 hours...")
last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
next_24h_predictions = []

for _ in range(24):
    next_pred = best_model_obj.predict(last_sequence, verbose=0)
    next_24h_predictions.append(next_pred[0, 0])
    
    # Update sequence (simplified - in production you'd update features too)
    last_sequence = np.roll(last_sequence, -1, axis=1)
    last_sequence[0, -1, 0] = next_pred[0, 0] / 25000  # Rough normalization

# Save predictions for dashboard
forecast_data = {
    'test_actual': y_test,
    'test_predictions': predictions[f'{best_model[0]} LSTM'],
    'next_24h': np.array(next_24h_predictions),
    'model_metrics': metrics[best_model[0]]
}

with open('forecast_results.pkl', 'wb') as f:
    pickle.dump(forecast_data, f)

print("💾 Forecast results saved for dashboard")
print("\n✅ MODEL TRAINING COMPLETE!")
print("\n🎯 Next step: Run the Streamlit dashboard to visualize your predictions!")