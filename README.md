# Physics-Informed Hybrid Modeling for Carbon Emissions Forecasting

## 🌍 Project Overview

This research project evaluates traditional time series models against advanced machine learning techniques for accurately predicting carbon emissions using Climate TRACE data for India. The study introduces a novel hybrid modeling approach combining Physics-Informed Neural Networks (PINNs) with classical and machine learning models to improve prediction accuracy across seven key sectors.

## 🎯 Motivation

Accurately predicting carbon emissions is vital for the world's progress towards climate goals and for companies working to meet ESG (Environmental, Social, and Governance) standards. This study addresses the limitations of traditional models in handling missing data (common in ESG reporting) and explores how hybrid approaches can capture both underlying emission dynamics and complex interactions.

## 🔬 Models & Algorithms Used

### **Classical Time Series Models**
- **ARIMA** - AutoRegressive Integrated Moving Average for trend analysis
- **SARIMA** - Seasonal ARIMA for capturing seasonal emission patterns  
- **Holt-Winters** - Exponential smoothing for trend and seasonality
- **Prophet** - Facebook's robust forecasting tool for emissions prediction

### **Machine Learning Models**
- **Random Forest** - Ensemble method for handling complex feature interactions
- **Gradient Boosting** - Sequential boosting for improved prediction accuracy
- **XGBoost** - Extreme Gradient Boosting with advanced optimization
- **SVR (Support Vector Regression)** - Non-linear regression with kernel methods

### **Novel Hybrid Approaches**
- **Physics-Informed Neural Networks (PINNs)** - Deep learning with physical constraints for emission patterns
- **PINN + Prophet** - Hybrid combining PINN main trends with Prophet residual correction
- **PINN + XGBoost** - Hybrid using XGBoost to model residual errors from PINN predictions

## 📊 Data & Methodology

### **Dataset**
- **Source**: Climate TRACE emissions data for India
- **Time Period**: January 2021 – May 2025
- **Training Data**: January 2021 – December 2023
- **Testing Data**: January 2024 – May 2025
- **Temporal Resolution**: Monthly aggregated data

### **Key Sectors Analyzed**
1. **Waste Management**
2. **Manufacturing**
3. **Fossil Fuels**
4. **Transportation**
5. **Power Generation**
6. **Agriculture**
7. **Buildings**

### **Data Preprocessing Pipeline**
- **Missing Value Handling**: Distribution-aware imputation techniques
- **Mixed-Type Columns**: Ordinal mapping for qualitative tags ("high", "very high")
- **Feature Engineering**: Lag features of past emissions for predictive modeling
- **Data Cleaning**: Outlier detection and temporal consistency validation

## 📈 Model Performance Results

| Model | MAPE (%) | Performance Rank |
|-------|----------|------------------|
| **PINN + Prophet** | **1.70%** | 🥇 **Best** |
| **Prophet (Standalone)** | 2.33% | 🥈 |
| **XGBoost** | 2.72% | 🥉 |
| **Gradient Boosting** | 2.72% | 🥉 |
| **Holt-Winters** | 2.76% | - |
| **SARIMA** | 3.67% | - |
| **PINN + XGBoost** | 4.33% | - |
| **PINN (Standalone)** | 8.18% | - |

### **Key Findings**
- ✅ **Hybrid PINN + Prophet** achieves lowest error (1.70% MAPE)
- ✅ **Hybrid approaches** significantly outperform standalone PINNs
- ✅ **Classical models** remain competitive (Holt-Winters: 2.76%)
- ✅ **Machine learning baselines** show consistent performance (~2.7% MAPE)

## 🛠️ Technical Architecture

### **Hybrid Modeling Framework**
```
Climate TRACE Data → Preprocessing → Feature Engineering
    ↓
PINN Training (Main Patterns) → Residual Calculation
    ↓
ML/Classical Model (Residual Correction) → Final Predictions
    ↓
Evaluation (MAPE) → Model Comparison
```

### **Implementation Structure**
```
├── data/
│   ├── climate_trace_india/        # Raw Climate TRACE data
│   ├── processed/                  # Cleaned and preprocessed data
│   └── sector_data/               # Seven-sector breakdown
├── src/
│   ├── preprocessing/
│   │   ├── data_cleaning.py       # Missing value handling
│   │   ├── feature_engineering.py # Lag features creation
│   │   └── imputation.py          # Distribution-aware imputation
│   ├── models/
│   │   ├── classical/             # ARIMA, SARIMA, Holt-Winters, Prophet
│   │   ├── machine_learning/      # RF, GB, XGBoost, SVR
│   │   ├── pinns/                 # Physics-Informed Neural Networks
│   │   └── hybrid/                # PINN + ML/Classical combinations
│   ├── evaluation/
│   │   ├── metrics.py             # MAPE calculation
│   │   └── model_comparison.py    # Performance benchmarking
│   └── utils/
│       ├── sector_analysis.py     # Seven-sector processing
│       └── visualization.py       # Results plotting
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── results/
│   ├── model_performance/         # MAPE scores and comparisons
│   ├── forecasts/                 # 2024-2025 predictions
│   └── sector_analysis/           # Per-sector results
└── config/
    ├── model_configs.yaml         # Hyperparameters
    └── data_config.yaml           # Data processing settings
```

## 🚀 Getting Started

### **Prerequisites**
```bash
Python 3.8+
TensorFlow/PyTorch (for PINNs)
scikit-learn
fbprophet
xgboost
pandas, numpy, matplotlib, seaborn
```

### **Installation**
```bash
# Clone repository
git clone <repository-url>
cd carbon-emissions-forecasting

# Install dependencies
pip install -r requirements.txt

# Install additional PINN dependencies
pip install tensorflow torch fbprophet xgboost climatetrace-api
```

### **Quick Start**
```python
from src.models.hybrid.pinn_prophet import PINNProphetHybrid
from src.preprocessing.data_cleaning import load_climate_trace_data

# Load India emissions data (Jan 2021 - May 2025)
data = load_climate_trace_data('data/climate_trace_india/')

# Initialize hybrid model
model = PINNProphetHybrid()

# Train on 2021-2023, test on 2024-2025
model.fit(data['2021':'2023'])
predictions = model.predict(data['2024':'2025'])

# Evaluate performance
mape = model.calculate_mape(predictions, actual)
print(f"PINN + Prophet MAPE: {mape:.2f}%")
```

## 🏭 Sector-Specific Analysis

The framework provides detailed analysis across seven key sectors:

- **Power Generation**: Captures seasonal demand patterns
- **Transportation**: Models traffic and fuel consumption trends  
- **Manufacturing**: Handles production cycle variations
- **Buildings**: Accounts for heating/cooling seasonality
- **Agriculture**: Models crop cycle emissions
- **Fossil Fuels**: Tracks extraction and processing patterns
- **Waste Management**: Captures disposal and treatment trends

## 🎯 Key Contributions

### **Methodological Innovations**
1. **Hybrid PINN Framework**: Novel combination of physics-informed modeling with classical/ML residual correction
2. **Sector-Specific Analysis**: Comprehensive evaluation across seven emission sectors  
3. **Missing Data Handling**: Distribution-aware imputation for ESG reporting challenges
4. **Mixed-Type Processing**: Ordinal mapping for qualitative emission indicators

### **Practical Applications**
- **ESG Compliance**: Enhanced emission forecasting for corporate sustainability
- **Policy Planning**: National-level insights for climate goal achievement
- **Industry Forecasting**: Sector-specific emission trend analysis
- **Carbon Transparency**: Improved accuracy for emission reporting

## 📊 Impact & Applications

This hybrid modeling framework demonstrates how national-level emissions data can be leveraged for industry-level forecasting, enabling organizations to:

- 🎯 **Anticipate emission trends** with 1.70% MAPE accuracy
- 📈 **Plan targeted reductions** based on sector-specific patterns  
- ✅ **Strengthen ESG compliance** through accurate forecasting
- 🌱 **Accelerate sustainable development** via AI-driven insights

## 📝 Citation

If you use this work, please cite:
```bibtex
@article{carbon_emissions_hybrid_2025,
  title={Physics-Informed Hybrid Modeling for Carbon Emissions Forecasting},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Climate TRACE** for comprehensive emissions data
- **Physics-Informed Neural Networks** research community
- **Facebook Prophet** and **XGBoost** development teams
- **ESG reporting standards** organizations

---

**Advancing carbon transparency through hybrid AI modeling for sustainable development** 🌱