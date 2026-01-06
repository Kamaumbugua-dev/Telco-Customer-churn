
# Telco Customer Churn Prediction App

## Overview

This project implements a **customer churn prediction system** using machine learning and provides an interactive **Streamlit dashboard** for real-time predictions.  

The app enables businesses to:

- Identify customers likely to churn.
- Evaluate churn risk and segment customers (Low, Medium, High).
- Gain actionable insights and recommendations to improve retention.

The system includes **model training, feature engineering, visualization**, and an interactive interface for predicting individual customer churn.

**Key Features:**

- Build, evaluate, and compare ML models: **Logistic Regression, Random Forest, XGBoost**
- Feature engineering and preprocessing for categorical and numerical variables
- Visualize churn trends, correlations, and feature importance
- Export predictions and metrics for reporting or Power BI integration
- Interactive Streamlit app with sliders and dropdowns for customer input
- Risk reasoning and plain-language explanations with actionable suggestions

**Dataset:** Telco Customer Churn (Kaggle)  
[Download link](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## Tech Stack

- **Python ** – Core programming language  
- **Streamlit ** – Interactive web app interface  
- **Pandas / NumPy** – Data manipulation and preprocessing  
- **Scikit-learn / XGBoost** – Machine learning models  
- **Matplotlib / Seaborn** – Data visualization  
- **Joblib** – Model serialization  

---

## Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd <project-folder>
````

2. **Create a virtual environment:**

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Required packages include:**
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `streamlit`, `joblib`

4. **Verify file structure:**

```
artifacts/
    model.pkl              # Pre-trained churn model
    feature_names.pkl      # Feature names for model
data/
    churn.csv              # Dataset for feature ranges
```

---

## Usage

1. **Run the Streamlit app:**

```bash
streamlit run app.py
```

2. **Features in the app:**

**Input Customer Data:**

* Numeric features (e.g., `tenure`, `MonthlyCharges`) via sliders
* Categorical features (e.g., `Contract`, `PaymentMethod`) via dropdowns

**Churn Prediction:**

* Predicts churn (`0 = stay`, `1 = leave`) using a pre-trained ML model
* Shows churn probability and risk segment (Low, Medium, High)

**Reasoning & Suggestions:**

* Provides rules-based explanations for predictions (e.g., short tenure, month-to-month contract)
* Suggests actionable retention strategies (e.g., loyalty offers, service upgrades)

**Plain-Language Summary:**

* Summarizes predictions, probability, risk, reasons, and suggested actions in a single readable block

---

## Model Training & Evaluation (Optional)

* Train **Logistic Regression, Random Forest, XGBoost** models using `train.py`
* Evaluate models using **Accuracy, Precision, Recall, F1-Score, and AUC**
* Automatically selects the best model based on highest AUC
* Tree-based models provide **feature importance visualizations**

---

## Project Structure

```
├── app.py                  # Streamlit application for predictions
├── train.py                # Model training and evaluation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── artifacts/
│   ├── model.pkl           # Pre-trained model
│   └── feature_names.pkl   # Model feature names
├── data/
│   └── churn.csv           # Original dataset for input options
├── best_model.pkl          # Saved best model
├── label_encoders.pkl      # Encoders for categorical features
├── scaler.pkl              # Scaler for numerical features
├── feature_columns.pkl     # Feature order for prediction
├── churn_predictions.csv   # Predicted churn and risk segments
├── model_metadata.pkl      # Model metadata and metrics
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Features Engineered

* **TenureGroup:** Customer lifetime segments
* **AvgMonthlyCharges:** Average monthly spend
* **HasMultipleServices:** Indicator for multiple services adoption
* **HasPhoneAndInternet:** Indicator for bundled services
* **PaperlessBilling:** Binary flag for paperless billing

**Additional Reasoning in the Streamlit App:**

* Month-to-month contract → higher churn risk
* Low tenure (<12 months) → higher churn risk
* No online security service → may reduce satisfaction
* Electronic check payment method → higher churn tendency
* High monthly charges → may cause dissatisfaction

---

## Business Insights

* Identifies **high, medium, and low churn risk** customers
* Calculates **potential revenue saved** through targeted retention campaigns
* Provides actionable suggestions for customer retention:

  * Loyalty benefits for short-tenure customers
  * Offering long-term contracts
  * Bundled services and online security packages
  * Encourage reliable payment methods

---

## Contribution

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Contact

**Developer:** Steven Kamau Mbugua
**Email:** [stevenk710@gmail.com](mailto:stevenk710@gmail.com)
**LinkedIn:** [linkedin.com/in/steven-kamau-mbugua](https://www.linkedin.com/in/steven-kamau-mbugua)

```


Do you want me to do that?
```
