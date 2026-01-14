# Airbnb Price Prediction & Investment Analysis

## 📊 Project Overview

This project analyzes Airbnb rental prices across three European cities (Berlin, Prague, Madrid) using machine learning to assess real estate investment profitability and the impact of amenities on rental income.

### Key Objectives
1. **Price Prediction**: Build ML models (OLS, XGBoost) to predict Airbnb rental prices
2. **Investment Analysis**: Calculate ROI and payback periods for property investments
3. **Amenity Impact**: Quantify how specific amenities (WiFi, dishwasher, etc.) affect rental prices
4. **City Comparison**: Compare investment opportunities across Berlin, Prague, and Madrid

---

## 🏗️ Project Structure

```
├── data/
│   ├── listings_Berlin.csv.gz
│   ├── listings_Prague.csv.gz
│   └── listings_Madrid.csv.gz
├── notebooks/
│   ├── berlin_analysis.ipynb
│   ├── prague_analysis.ipynb
│   └── madrid_analysis.ipynb
├── models/
│   ├── xgboost_model.json
│   └── shap_values/
├── results/
│   ├── feature_importance.xlsx
│   └── investment_recommendations.xlsx
└── README.md
```

---

## 🔧 Technologies & Libraries

- **Python 3.10+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: 
  - statsmodels (OLS regression)
  - xgboost (gradient boosting)
  - scikit-learn (preprocessing, metrics)
- **Visualization**: matplotlib, seaborn, plotly
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **EDA**: SweetViz

---

## 📈 Methodology

### 1. Data Preparation

```python
# Key steps:
- Load Airbnb listings from InsideAirbnb.com
- Filter: entire home/apt only
- Remove outliers (5th-95th percentile)
- Handle missing values
- Feature engineering:
  - Distance from city center
  - Neighborhood encoding
  - Amenity binary features (70+ amenities)
  - Property type classification
```

### 2. Feature Engineering

**Created Variables:**
- `dist_from_center`: Euclidean distance from Brandenburg Gate (Berlin)
- `neighbourhood_cleansed_map`: Top 20 neighborhoods + "other"
- `property_type_map`: Categorized property types
- `is_[amenity]`: Binary indicators for 70+ amenities (WiFi, dishwasher, etc.)
- `n_amenities`: Total amenity count

**Investment Metrics:**
```python
# Assumptions (Berlin example):
m2_price = 5000 + (1 - dist_from_center) * 3000  # €5000-8000/m²
area_m2 = 20 + bedrooms*10 + bathrooms*5
apt_purchase_price = area_m2 * m2_price
payback_time = apt_purchase_price / price_predictions  # days
```

### 3. Model Training

#### OLS (Ordinary Least Squares)
```python
model = smf.ols('''
    price ~ accommodates + bathrooms + bedrooms + beds
          + is_wifi + is_tv + is_dishwasher
          + dist_from_center + neighbourhood_cleansed_map
          + property_type_map
''', data=data).fit()
```

**Results (Berlin):**
- **RMSE**: 47.07 €
- **MAE**: 34.59 €
- **R²**: 0.46

#### XGBoost
```python
model = xgboost.XGBRegressor(
    n_estimators=200,
    verbosity=1
)
model.fit(X_train, y_train)
```

**Results (Berlin, validation set):**
- **RMSE**: 42.26 € (↓10% vs OLS)
- **MAE**: 32.02 € (↓7% vs OLS)
- Training RMSE: 5.30 € (some overfitting detected)

---

## 🎯 Key Findings

### Feature Importance (SHAP Analysis)

**Top 10 Price Drivers (Berlin):**

| Feature | Impact | Description |
|---------|--------|-------------|
| `property_type_serviced_apt` | +31.2% | Serviced apartments command premium |
| `accommodates` | +5.5% | Guest capacity strongly linked to price |
| `bedrooms` | +4.9% | +12.95€ per bedroom |
| `bathrooms` | +5.0% | +21.13€ per bathroom |
| `is_self_check_in` | +4.9% | +8.13€ (0→1) |
| `is_dishwasher` | +4.0% | +5.64€ (0→1) |
| `dist_from_center` | -3.4% | -106€ per unit distance |
| `is_fire_extinguisher` | +3.4% | +11.14€ (0→1) |
| `neighbourhood_Prenzlauer_Berg` | +4.5% | +6.25€ premium |
| `n_amenities` | +1.6% | +0.30€ per amenity |

### Amenity ROI Analysis

**High-Impact Amenities (Berlin):**
```
✅ Dishwasher:        +5.64€/night → +2,059€/year
✅ Self Check-in:     +8.13€/night → +2,967€/year
✅ Fire Extinguisher: +11.14€/night → +4,066€/year
✅ TV:                +3.16€/night → +1,153€/year
✅ Patio/Balcony:     +2.77€/night → +1,011€/year
```

**Regression Analysis (Continuous Variables):**
```
bedrooms:        +30.57€ per bedroom (R²=0.195)
accommodates:    +14.63€ per guest capacity (R²=0.247)
bathrooms:       +63.30€ per bathroom (R²=0.133)
n_amenities:     +1.32€ per amenity (R²=0.082)
dist_from_center: -166.72€ per unit (R²=0.017)
```

### Investment Metrics (Berlin)

**Average Property:**
- Purchase Price: 218,850€
- Predicted Daily Rate: 133€
- Payback Period: 1,810 days (~5 years)

**Best ROI Neighborhoods:**
1. Prenzlauer Berg Südwest: 173€/night
2. Brunnenstr. Süd: 171€/night
3. Alexanderplatz: 160€/night

---

## 📊 Model Performance Comparison

| Metric | OLS | XGBoost | Winner |
|--------|-----|---------|--------|
| **RMSE (validation)** | 47.07€ | 42.26€ | **XGBoost** ✓ |
| **MAE (validation)** | 34.59€ | 32.02€ | **XGBoost** ✓ |
| **Interpretability** | High | Medium (SHAP) | **OLS** ✓ |
| **Normality assumption** | Yes | No | **XGBoost** ✓ |
| **Feature interactions** | Limited | Automatic | **XGBoost** ✓ |
| **Training time** | Fast | Moderate | **OLS** ✓ |

---

## 🔍 SHAP Analysis Highlights

### Waterfall Plot Interpretation
For a typical property (140€/night prediction):
- Base value (E[f(X)]): 139.96€
- Bedrooms (2): +25.89€
- Accommodates (4): +22.47€
- Location premium: +6.25€
- Amenities boost: +8.13€
- **Final prediction**: ~142.70€

### Feature Interactions
- **Bedrooms × Bathrooms**: Strong positive correlation (0.70)
- **Distance × Neighborhood**: Location hierarchy matters more than raw distance
- **Amenities × Property Type**: Serviced apartments benefit more from premium amenities

---

## 💡 Investment Recommendations

### For Berlin:
1. **Target neighborhoods**: Prenzlauer Berg Südwest, Brunnenstr. Süd
2. **Optimal property**: 2BR/1BA, 4-5 guests, ~45m²
3. **Must-have amenities**: WiFi, dishwasher, self-check-in, fire safety
4. **Expected ROI**: 5-6 year payback at 70% occupancy

### Quick Wins (Low-Cost, High-Impact):
```python
Investment    Cost      Annual Return   Payback
-------------------------------------------------
Self check-in  100€     +2,967€        1.2 months
Fire alarm     50€      +4,066€        0.4 months
Dishwasher     400€     +2,059€        2.3 months
TV             200€     +1,153€        2.1 months
```

---

## 🚀 Usage

### Setup
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn plotly
```

### Run Analysis
```python
# Load and prepare data
data = pd.read_csv('listings_Berlin.csv.gz')
data = prepare_data(data)  # Apply preprocessing

# Train XGBoost
model = train_xgboost(X_train, y_train)

# SHAP analysis
explainer = shap.Explainer(model.predict, X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# Calculate investment metrics
results = calculate_investment_metrics(data, predictions)
```

---

## 📁 Data Source

**Inside Airbnb** (http://insideairbnb.com/get-the-data)
- Berlin: September 2025 snapshot
- Prague: [Add date]
- Madrid: [Add date]

**Key variables:**
- `price`: Nightly rate (€)
- `bedrooms`, `bathrooms`, `accommodates`: Property specs
- `amenities`: JSON list of 250+ possible amenities
- `latitude`, `longitude`: Geolocation
- `neighbourhood_cleansed`: Standardized neighborhoods

---

## ⚠️ Limitations

1. **Model Assumptions**:
   - Linear m²/price relationship (€5,000-8,000/m²)
   - Fixed apartment size formula (20 + bedrooms×10 + bathrooms×5)
   - Does not account for seasonal pricing variations

2. **Data Quality**:
   - 35% missing price data removed
   - Outliers (>95th percentile) excluded
   - Self-reported amenities may have inconsistencies

3. **Generalization**:
   - Models trained on 60% data (stratified by price quartiles)
   - Performance may vary for luxury/budget segments
   - Local regulations not factored into ROI calculations

---

## 🔮 Future Work

- [ ] Time-series analysis for seasonal pricing
- [ ] Multi-city comparative model
- [ ] Deep learning (neural networks) for better predictions
- [ ] Scraping real estate prices for accurate purchase costs
- [ ] Occupancy rate modeling
- [ ] Regulatory compliance checker

---

**Last Updated**: January 2026  
**Status**: ✅ Complete (Berlin) | 🚧 In Progress (Prague, Madrid)
