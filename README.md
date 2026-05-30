# Nike-Sales-Data-Cleaning-EDA-Profit-Prediction
nike sales analyst
# 👟 Nike Sales Data Cleaning, EDA & Profit Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-150458?style=flat-square&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat-square&logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-teal?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Author](https://img.shields.io/badge/Author-Samwel%20Morara-purple?style=flat-square)

---

## 📌 Project Overview

This project covers a **complete end-to-end data science pipeline** on a real-world, messy Nike sales dataset sourced from Kaggle. The dataset contained severe quality issues — missing values across multiple columns, inconsistent date formats, and negative unit sold entries — making data cleaning the most critical first step before any analysis or modelling could begin.

The project is structured across four major phases:
1. **Data Cleaning & Preparation** — handling nulls, inconsistencies, and type conversion
2. **Exploratory Data Analysis (EDA)** — uncovering patterns in revenue, profit, regions, and product lines
3. **Feature Engineering** — creating time-based and pricing features for modelling
4. **Machine Learning** — predicting profit using Random Forest with GridSearchCV tuning

---

## 📂 Dataset

| Property | Detail |
|---|---|
| Source | Kaggle — `nayakganesh007/nike-sales-uncleaned-dataset` |
| File | `Nike_Sales_Uncleaned.csv` |
| Raw Records | 2,500 |
| Clean Records | 600 (after dropping unparseable dates) |
| Columns | 13 (Order_ID, Gender_Category, Product_Line, Product_Name, Size, Units_Sold, MRP, Discount_Applied, Revenue, Order_Date, Sales_Channel, Region, Profit) |

### Raw Data Quality Issues

| Column | Missing Values | Fix Applied |
|---|---|---|
| `Discount_Applied` | 1,668 | Imputed with 0 (no discount assumed) |
| `MRP` | 1,254 | Imputed with median |
| `Units_Sold` | 1,235 | Imputed with median |
| `Order_Date` | 616 initially, 1,900 after parsing | Rows dropped (unparseable formats) |
| `Size` | 510 | Imputed with mode (`'L'`) |

---

## 🔬 Methodology

### Phase 1 — Data Cleaning

- Loaded dataset via `kagglehub` into a Pandas DataFrame
- Converted `Order_Date` to datetime using `errors='coerce'` to catch all inconsistent formats (e.g. `04-10-2024`, `2024/09/12`)
- After conversion, 1,900 rows had `NaT` values and were dropped, reducing the dataset from 2,500 to **600 clean records**
- Imputed `Size` with mode (`'L'`), `Units_Sold` and `MRP` with their medians, and `Discount_Applied` with `0`
- Confirmed zero duplicates across all 600 records
- Converted categorical columns to `category` dtype for memory efficiency — reduced memory from 254 KB to 42.9 KB
- Converted `Units_Sold` from `float64` to `int64`

### Phase 2 — Outlier Detection

- Applied IQR method to `Revenue` column
- Initial IQR flagged all non-zero revenue as outliers (because 75% of revenue values were 0)
- Adjusted by filtering for non-zero revenue first, then re-applying IQR — found **2 true outliers**: Order 2248 (Revenue: 25,079) and Order 4008 (Revenue: 21,587)

### Phase 3 — Exploratory Data Analysis (EDA)

Investigated distributions and group-level patterns across the cleaned dataset:

| Analysis | Key Finding |
|---|---|
| Revenue Distribution | Highly skewed — majority of entries have Revenue = 0 with occasional high-value spikes |
| Gender Category | Kids most frequent (205/600), followed by Men and Women |
| Product Line | Training most common (129/600) |
| Sales Channel | 50/50 split between Online and Retail (300 each) |
| Top Region | Kolkata (123 orders) |
| Revenue by Product Line | Training leads; Running shows negative total revenue |
| Daily Revenue Trend | Highly volatile with frequent zero-revenue days |

### Phase 4 — Feature Engineering

Created new features from `Order_Date` and pricing columns:

```python
# Time-based features
df['Year']         = df['Order_Date'].dt.year
df['Month']        = df['Order_Date'].dt.month
df['Day_of_Week']  = df['Order_Date'].dt.dayofweek
df['Day_of_Year']  = df['Order_Date'].dt.dayofyear
df['Week_of_Year'] = df['Order_Date'].dt.isocalendar().week.astype(int)
df['Is_Weekend']   = df['Order_Date'].dt.dayofweek.isin([5, 6]).astype(int)

# Pricing features
df['Effective_Price_per_Unit'] = df['MRP'] * (1 - df['Discount_Applied'])
```

### Phase 5 — Data Preparation for ML

- **Target variable:** `Profit`
- **Dropped columns:** `Order_ID`, `Order_Date`, `Revenue` (avoids data leakage), `Price_per_Unit`
- **One-Hot Encoding** via `pd.get_dummies` on 6 categorical columns → expanded features from 16 to **53**
- **StandardScaler** applied to all numerical features (mean=0, std=1)
- **Train/Test Split:** 80/20 → 480 training, 120 test records (`random_state=42`)

### Phase 6 — Model Training & Evaluation

**Initial Random Forest:**
```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
```

| Metric | Initial Model | Tuned Model |
|---|---|---|
| MAE | 1,397.61 | 1,353.87 |
| MSE | 2,618,954.90 | 2,415,652.40 |
| RMSE | 1,618.32 | 1,554.24 |
| R² | −0.13 | −0.05 |

**GridSearchCV Tuning:**
```python
param_grid = {
    'n_estimators':      [50, 100, 200],
    'max_features':      ['sqrt', 'log2'],
    'max_depth':         [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2]
}
# 72 combinations × 3-fold CV = 216 total fits
```

**Best Parameters Found:**
```
max_depth=10, max_features='log2', min_samples_leaf=1,
min_samples_split=2, n_estimators=200
```

---

## 📊 Top 10 Feature Importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | Day_of_Year | 0.1224 |
| 2 | Effective_Price_per_Unit | 0.0966 |
| 3 | MRP | 0.0703 |
| 4 | Week_of_Year | 0.0639 |
| 5 | Day_of_Week | 0.0556 |
| 6 | Units_Sold | 0.0488 |
| 7 | Discount_Applied | 0.0374 |
| 8 | Year | 0.0242 |
| 9 | Region_Pune | 0.0211 |
| 10 | Size_XL | 0.0210 |

---

## 💡 Key Insights & Interpretation

**Why is R² negative?**
A negative R² means the model performs worse than simply predicting the mean profit for every entry. This is an important and honest finding — it tells us that `Profit` in this dataset is not well explained by the available features alone. The likely causes are:

- Revenue being zero for the majority of entries (a data collection anomaly)
- Profit appears to be driven by factors not captured in the dataset (e.g. cost of goods, promotions, stock levels)
- The 600-record cleaned dataset is relatively small for 53 features after encoding

**What the model did learn:**
- Time-based features (Day of Year, Week of Year) are the strongest signals — suggesting **seasonality** matters
- Pricing features (Effective Price per Unit, MRP) are the next strongest — higher-priced items tend to influence profit more
- Regional and size effects exist but are weak individually

**Business Recommendations:**
- Investigate the large number of zero-revenue entries — these may represent returns, cancelled orders, or data pipeline gaps
- Running product line shows negative total revenue — warrants urgent investigation
- Focus marketing efforts around seasonal peaks identified by the time-series analysis
- Kolkata and Online channel show strongest performance — prioritise these in campaigns

---

## 🔮 Future Improvements

- [ ] Investigate and resolve zero-revenue entries before modelling
- [ ] Try predicting `Units_Sold` as target (less noisy than `Profit`)
- [ ] Add external features: promotional calendars, competitor pricing, inventory data
- [ ] Apply SHAP values for deeper feature explanation
- [ ] Try XGBoost and LightGBM and compare with Random Forest
- [ ] Build an interactive sales dashboard using Plotly or Streamlit
- [ ] Deploy best model as a REST API using Flask

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Pandas | Data loading, cleaning, feature engineering |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | EDA visualisations |
| Scikit-Learn | ML pipeline, scaling, GridSearchCV |
| kagglehub | Dataset loading from Kaggle |
| Jupyter Notebook | Interactive development |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/onguti-star/data-science-projects.git
cd data-science-projects/nike-sales
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the notebook
```bash
jupyter notebook nike_sales.ipynb
```

### requirements.txt
```
pandas
numpy
matplotlib
seaborn
scikit-learn
kagglehub
jupyter
```

---

## 📁 Project Structure

```
nike-sales/
│
├── nike_sales.ipynb        ← Main notebook (cleaning + EDA + ML)
├── README.md               ← This file
└── requirements.txt        ← Python dependencies
```

---

## 👤 Author

**Samwel Morara Onguti**
- 📧 Email: ongutimorara776@gmail.com
- 🌐 Portfolio: [manuallabor.gt.tc](https://manuallabor.gt.tc)
- 💼 GitHub: [github.com/onguti-star](https://github.com/onguti-star)
- 📍 Nairobi, Kenya
- 🎓 B.Sc. Mathematics & Computer Science — Multimedia University of Kenya (Expected 2027)
- 📜 ALX Africa Certified Data Scientist (2025–2026)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built as part of ALX Africa Data Scientist certification — 2025/2026*
