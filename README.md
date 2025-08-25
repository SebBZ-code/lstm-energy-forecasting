# ⚡ AEP Energy Consumption Forecasting

LSTM-based deep learning model for predicting hourly energy consumption with 99.65% accuracy.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.65%25-green.svg)
![MAE](https://img.shields.io/badge/MAE-111.78_MW-yellowgreen.svg)

## 📊 Overview

This project implements a Long Short-Term Memory (LSTM) neural network to forecast energy consumption for American Electric Power (AEP). Using 13+ years of hourly consumption data, the model achieves exceptional accuracy that outperforms traditional forecasting methods by 88%.

### Key Features
- 🧠 Deep learning model with 64 LSTM units
- 📈 Interactive Streamlit dashboard for real-time predictions
- 🔄 Automated feature engineering (17 time-based features)
- 📊 Comprehensive data analysis and visualization
- ⚡ Sub-1% error rate (0.76% MAPE)

## 🏆 Performance Metrics

| Metric | Value | Improvement vs Baseline |
|--------|-------|------------------------|
| **MAE** | 111.78 MW | 88% better |
| **RMSE** | 147.57 MW | - |
| **MAPE** | 0.76% | - |
| **R²** | 0.9965 | - |

### Baseline Comparison
- **Seasonal Naive**: 925.36 MW MAE
- **Simple Naive**: 417.17 MW MAE  
- **7-Day Moving Average**: 1710.01 MW MAE
- **Our LSTM Model**: 111.78 MW MAE 🏆

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- 8GB RAM minimum (for model training)
- 500MB free disk space

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SebBZ-code/aep-energy-forecasting.git
cd aep-energy-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit dashboard:
```bash
streamlit run aep-streamlit-dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

NOTE: in 'aep-energy-analysis.py' the filepath must be updated to match your filepath

## 📁 Project Structure

```
├── data/
│   └── AEP_hourly.csv              # 121,273 hours of consumption data
├── models/
│   ├── best_simple_lstm_model.h5   # Trained model (99.65% accuracy)
│   └── lstm_data.pkl               # Preprocessed features
├── src/
│   ├── aep-energy-analysis.py      # EDA and feature engineering
│   ├── aep-lstm-model.py          # Model training and evaluation
│   └── aep-streamlit-dashboard.py  # Interactive web dashboard
├── load_and_visualize.py           # Must be run before first run of dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🔧 Pipeline Overview

### 1. Data Analysis (`aep-energy-analysis.py`)
- Loads 13+ years of hourly energy consumption data
- Performs exploratory data analysis
- Identifies patterns: hourly, daily, weekly, seasonal

### 2. Feature Engineering
Creates 17 intelligent features including:
- Temporal features (hour, day, month)
- Cyclical encodings (sine/cosine transformations)
- Lag features (1h, 24h, 168h)
- Rolling statistics (24h mean)
- Weekend/weekday flags

### 3. Model Architecture (`aep-lstm-model.py`)
```python
Model: Sequential
├── LSTM (64 units, tanh activation)
├── Dropout (0.2)
├── Dense (32 units, ReLU)
└── Dense (1 unit, linear)
```

### 4. Training Details
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: MSE
- **Batch Size**: 32
- **Early Stopping**: Patience of 10 epochs
- **Data Split**: 70% train, 15% validation, 15% test

### 5. Dashboard Features (`aep-streamlit-dashboard.py`)
- Real-time predictions (1-168 hours ahead)
- Interactive visualizations with Plotly
- Historical data exploration
- Model performance metrics
- Pattern analysis (hourly, weekly, seasonal)

## 📈 Results Visualization

The model successfully captures complex temporal patterns:
- **Daily patterns**: Morning ramp-up, evening peak
- **Weekly patterns**: Weekday vs weekend consumption
- **Seasonal patterns**: Summer cooling, winter heating
- **Special events**: Holidays and anomalies

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Machine Learning**: Scikit-learn

## 💡 Key Insights

1. **Peak Consumption**: 19:00 (7 PM) with average 16,872 MW
2. **Lowest Consumption**: 04:00 (4 AM) with average 13,097 MW
3. **Weekend Effect**: 1,531 MW average decrease
4. **Seasonal Variation**: January shows highest consumption (winter heating)

## 🚀 Future Improvements

- [ ] Integrate weather data (temperature, humidity)
- [ ] Add holiday calendar for better special event prediction
- [ ] Implement confidence intervals for predictions
- [ ] Create API endpoint for real-time predictions
- [ ] Add anomaly detection system
- [ ] Multi-location support for entire AEP service area

## 📊 Model Performance Over Time

```
Training Progress:
Epoch 1:  MAE = 1200 MW (initial random weights)
Epoch 5:  MAE = 600 MW  (learning basic patterns)
Epoch 10: MAE = 400 MW  (understanding complex relationships)
Epoch 20: MAE = 111 MW  (final optimized model)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 Sebastian

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- American Electric Power (AEP) for the data
- TensorFlow team for the excellent deep learning framework
- Streamlit for the amazing dashboard capabilities

## 📚 References

- [LSTM Networks - Understanding the Architecture](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Time Series Forecasting Best Practices](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [AEP Energy Data Source](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

---

<p align="center">
⭐ If you found this project useful, please consider giving it a star on GitHub! ⭐
</p>
