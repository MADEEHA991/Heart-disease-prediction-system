# ❤️ Heart Disease Prediction System

A full-stack machine learning web application that predicts the risk of heart disease from clinical patient data. Built entirely in Python using **Streamlit** for the frontend and **scikit-learn** for the ML backend — no separate server required.

---

## 📂 Dataset

**UCI Heart Disease Dataset (Kaggle mirror)**

- 🔗 [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **File used:** `heart.csv` (must be in the project root)
- **Rows:** 1,025 patient records
- **Features:** 13 clinical attributes
- **Target:** `0` = No Heart Disease, `1` = Heart Disease Present

### Feature Descriptions

| Feature    | Description                              | Type        |
|------------|------------------------------------------|-------------|
| `age`      | Age in years                             | Numeric     |
| `sex`      | Sex (1 = Male, 0 = Female)               | Binary      |
| `cp`       | Chest pain type (0–3)                    | Categorical |
| `trestbps` | Resting blood pressure (mm Hg)           | Numeric     |
| `chol`     | Serum cholesterol (mg/dl)                | Numeric     |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1/0)   | Binary      |
| `restecg`  | Resting ECG results (0–2)                | Categorical |
| `thalach`  | Maximum heart rate achieved              | Numeric     |
| `exang`    | Exercise-induced angina (1/0)            | Binary      |
| `oldpeak`  | ST depression induced by exercise        | Numeric     |
| `slope`    | Slope of peak exercise ST segment (0–2) | Categorical |
| `ca`       | Number of major vessels colored (0–3)   | Numeric     |
| `thal`     | Thalassemia type (0–3)                   | Categorical |

---

## 📋 Project Description

This application provides an end-to-end heart disease prediction pipeline:

1. **Data Loading** — reads `heart.csv` directly from the project folder  
2. **Model Training** — trains and evaluates 5 ML classifiers automatically on first run  
3. **Auto Model Selection** — picks the best model by ROC-AUC score  
4. **Interactive Frontend** — a 4-page Streamlit app for exploration, evaluation, and prediction  

### Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, dataset summary, feature reference table |
| 📊 Data Explorer | Raw data preview, distributions, correlation heatmap |
| 🤖 Model Performance | Leaderboard, ROC curves, confusion matrix, feature importances, cross-validation |
| 🩺 Predict | Enter patient data → instant risk probability + gauge chart |

---

## 🛠️ Technologies Used

| Layer | Library | Purpose |
|-------|---------|---------|
| Frontend | `streamlit 1.35` | UI, layout, forms, charts |
| Data | `pandas 2.2` | Data loading and manipulation |
| Numerics | `numpy 1.26` | Array operations |
| ML Backend | `scikit-learn 1.5` | Model training, scaling, evaluation |
| Visualization | `plotly 5.22` | Interactive charts and gauge |
| Visualization | `matplotlib 3.9` | Heatmap rendering |
| Visualization | `seaborn 0.13` | Seaborn heatmap styling |
| Persistence | `joblib 1.4` | Saving trained model and scaler |

**ML Models trained and compared:**
- Random Forest Classifier
- Gradient Boosting Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd heart-disease-prediction
```

Or simply place `app.py`, `requirements.txt`, and `heart.csv` in the same folder.

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> **Note:** The first run trains all 5 models — this takes ~15–30 seconds.  
> Subsequent runs use Streamlit's `@st.cache_resource` cache and load instantly.

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── app.py               # Full application — frontend + backend (all-in-one)
├── requirements.txt     # Python dependency list
├── heart.csv            # Dataset (must be present in root)
├── README.md            # This file
│
└── (auto-generated on first run)
    ├── heart_model.joblib   # Saved best ML model
    └── heart_scaler.joblib  # Saved StandardScaler
```

---

## 🔑 Key Information

| Item | Detail |
|------|--------|
| Language | Python 3.9+ |
| Framework | Streamlit (frontend + server) |
| Dataset size | 1,025 rows × 14 columns |
| Train / Test split | 80% / 20% (stratified) |
| Feature scaling | StandardScaler (zero mean, unit variance) |
| Model selection | Best ROC-AUC on test set |
| Expected best accuracy | ~98–99% (Random Forest / Gradient Boosting) |
| Expected best ROC-AUC | ~0.99 |

---

## ⚕️ Medical Disclaimer

> This tool is intended for **educational and research purposes only**.  
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
> Always seek the advice of a qualified healthcare provider.

---

*Built with ❤️ using Python, Streamlit & scikit-learn*
