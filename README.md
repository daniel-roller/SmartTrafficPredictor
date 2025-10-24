# SmartTrafficPredictor

A comprehensive traffic flow prediction system that uses machine learning to forecast traffic patterns based on historical data.

## 🚀 Features

- **Multiple ML Models**: XGBoost, Random Forest, and SVM
- **Advanced Feature Engineering**: Peak hours, holidays, weekends, time-based features
- **Comprehensive Evaluation**: Multiple metrics (MAE, RMSE, MAPE, R²) with visual comparisons
- **Complete Pipeline**: From data loading to model evaluation
- **Modular Design**: Easy to extend and customize

## 📁 Project Structure

```
SmartTrafficPredictor/
├── src/
│   ├── config.py              # Configuration settings
│   ├── utils.py               # Utility functions (plotting, metrics)
│   ├── data_loader.py         # Data loading and integration
│   ├── preprocess.py          # Data cleaning and preprocessing
│   ├── feature_engineering.py # Traffic feature creation
│   ├── train_models.py        # Model training (XGBoost, RF, SVM)
│   ├── evaluate.py            # Model evaluation and comparison
│   └── main.py                # Main workflow integration
├── data/
│   ├── raw/                   # Raw CSV data files
│   └── processed/             # Processed data files
├── models/                    # Saved trained models
├── results/                   # Evaluation results and plots
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/daniel-roller/SmartTrafficPredictor.git
cd SmartTrafficPredictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Quick Start (with sample data)

Run the complete pipeline with sample data:

```bash
cd src
python main.py --sample-data --num-samples 10000
```

### Using Your Own Data

1. Place your CSV files in the `data/raw/` directory
2. Ensure your data has a `timestamp` column and a `traffic_flow` column
3. Run the pipeline:

```bash
cd src
python main.py
```

### Advanced Options

```bash
# With all features (lag and rolling statistics)
python main.py --sample-data --include-lag --include-rolling

# Skip training and use existing models
python main.py --skip-training

# With data normalization
python main.py --sample-data --normalize

# Custom sample data parameters
python main.py --sample-data --num-samples 20000 --start-date 2024-01-01 --freq 30min
```

### Running Individual Modules

Each module can be run independently for testing:

```bash
# Test data loader
python data_loader.py

# Test preprocessor
python preprocess.py

# Test feature engineering
python feature_engineering.py

# Test model training
python train_models.py

# Test evaluation
python evaluate.py
```

## 📈 Output

The system generates the following outputs:

### Data Files (in `data/processed/`)
- `traffic_data.csv`: Original or sample data
- `cleaned_traffic_data.csv`: Preprocessed data
- `featured_traffic_data.csv`: Data with engineered features

### Model Files (in `models/`)
- `xgboost_model.pkl`: Trained XGBoost model
- `random_forest_model.pkl`: Trained Random Forest model
- `svm_model.pkl`: Trained SVM model
- `feature_columns.pkl`: Feature column names

### Results (in `results/`)
- `model_comparison.png`: Visual comparison of all models
- `evaluation_results.txt`: Detailed metrics for each model
- `evaluation_report.txt`: Comprehensive evaluation report
- `xgboost_predictions.png`: XGBoost prediction visualization
- `random_forest_predictions.png`: Random Forest prediction visualization
- `svm_predictions.png`: SVM prediction visualization

## 🎯 Features

### Data Loading
- Loads multiple CSV files from the raw data directory
- Can generate sample data for testing
- Integrates data from multiple sources

### Preprocessing
- Handles missing values (interpolation, filling, or dropping)
- Removes outliers using IQR or Z-score methods
- Optional data normalization (standard or min-max)

### Feature Engineering
- **Time Features**: Hour, day of week, month, day of year
- **Cyclical Features**: Sine/cosine encoding for time periodicity
- **Weekend Indicator**: Binary feature for weekends
- **Peak Hour Indicator**: Morning (7-9) and evening (17-19) peaks
- **Holiday Indicator**: Customizable holiday list
- **Optional Lag Features**: Previous time period values
- **Optional Rolling Features**: Moving averages and statistics

### Model Training
- **XGBoost**: Gradient boosting with tree-based learning
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support Vector Machine regression

### Evaluation
- **Metrics**: MAE, RMSE, MAPE, R²
- **Visualizations**: Time series plots, scatter plots, comparison charts
- **Comprehensive Reports**: Best model recommendations

## ⚙️ Configuration

Edit `src/config.py` to customize:

- Data paths and column names
- Peak hour definitions
- Holiday list
- Data split ratios
- Model parameters
- Evaluation metrics

## 📝 Example Configuration

```python
# Peak hours (24-hour format)
MORNING_PEAK_START = 7
MORNING_PEAK_END = 9
EVENING_PEAK_START = 17
EVENING_PEAK_END = 19

# Holidays (MM-DD format)
HOLIDAYS = ['01-01', '05-01', '10-01', '12-25']

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

## 🔍 Model Comparison

The system automatically compares all models and identifies the best performer based on multiple metrics. Results include:

- Visual comparison charts
- Detailed metric tables
- Best model recommendations
- Performance rankings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

- Daniel Roller

## 🙏 Acknowledgments

- Built for traffic flow prediction and analysis
- Designed for easy extension and customization
- Suitable for both research and production use