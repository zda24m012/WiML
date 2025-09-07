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


## 🎯 Key Contributions

### **Methodological Innovations**
1. **Hybrid PINN Framework**: Novel combination of physics-informed modeling with classical/ML residual correction
2. **Sector-Specific Analysis**: Comprehensive evaluation across seven emission sectors  
3. **Missing Data Handling**: Distribution-aware imputation for ESG reporting challenges
4. **Mixed-Type Processing**: Ordinal mapping for qualitative emission indicators

## 📊 Impact & Applications

This hybrid modeling framework demonstrates how national-level emissions data can be leveraged for industry-level forecasting, enabling organizations to:

- 🎯 **Anticipate emission trends** with 1.70% MAPE accuracy
- 📈 **Plan targeted reductions** based on sector-specific patterns  
- ✅ **Strengthen ESG compliance** through accurate forecasting
- 🌱 **Accelerate sustainable development** via AI-driven insights
