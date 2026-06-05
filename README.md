# 🏙️ Tehran Housing Price Prediction

> Predicting residential property prices in Tehran using XGBoost — based on web-scraped real estate data.

---

## 📌 Overview

This project builds an XGBoost regression model to predict housing prices (in USD) in Tehran using real estate listings scraped from the web. It covers the full prediction pipeline: data cleaning, outlier removal, feature preprocessing, model training, and evaluation.



## 🗂️ Dataset

- **Source:** Web-scraped real estate listings
- **Target variable:** `Price(USD)`
- **Features:**

| Feature | Type | Description |
|---|---|---|
| `Area` | Numeric | Property size in m² |
| `Room` | Numeric | Number of rooms |
| `Parking` | Binary | Parking available (1/0) |
| `Warehouse` | Binary | Warehouse available (1/0) |
| `Elevator` | Binary | Elevator available (1/0) |
| `Address` | Categorical | Tehran neighborhood |

> ⚠️ Raw data is not included in this repo due to scraping constraints.

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/your-username/tehran-housing-price-prediction.git
cd tehran-housing-price-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook
```

**Requirements:** Python 3.8+, xgboost, scikit-learn, pandas, numpy, matplotlib, jupyter

---

## 🔬 Project Pipeline

1. **Data Loading** — Read scraped CSV, parse and cast feature types
2. **EDA** — Distribution histograms for all features + correlation analysis with `Price(USD)`
3. **Outlier Removal** — IQR-based filtering on `Area` (4×IQR) and `Price(USD)` (5×IQR)
4. **Preprocessing** — `StandardScaler` on numeric features, `OneHotEncoder` on `Address`
5. **Modeling** — XGBoost regressor inside a `sklearn` Pipeline
6. **Evaluation** — RMSE and R² score on a 20% test split

---

## 🤖 Model Configuration

```python
xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=8,
    learning_rate=0.05,
    n_estimators=200,
    eval_metric='rmse'
)
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| RMSE | — |
| R² | — |

> Results will be updated after final model run.

---

## 🛠️ Tech Stack

- Python 3.x
- XGBoost
- scikit-learn (Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder)
- pandas / numpy
- matplotlib
- Jupyter Notebook

---

## 📬 Contact

Open an issue or reach out for questions, feedback, or dataset requests.
