# Global Health & Development ML Analytics
### *Predictive Modeling of Life Expectancy and Economic Development Class using Advanced Regression, Imputation, and Classification Pipelines*

This repository implements a machine learning and statistical modeling pipeline to analyze global health, social, and economic indicators. Using a dataset of country-level indicators from 2000 to 2015, this project builds regression pipelines to predict **Life Expectancy** and classification models to identify **Economic Development Status** (Developing vs. Developed). 

Additionally, it incorporates a cross-feature regression imputation method to predict and reconstruct future feature values (for years 2013–2015) using historical data, evaluating the downstream impacts of feature imputation on prediction metrics.

---

## 📂 Repository Structure

```
health-development-ml/
├── data/
│   └── Life-Expectancy-Data-Updated.csv   # Unified country indicators dataset (2000-2015)
├── src/
│   ├── life_expectancy_regression.py      # Single, multiple, and polynomial regression modeling
│   ├── life_expectancy_imputation.py      # Feature imputation + life expectancy prediction
│   ├── economic_classification.py         # Single, polynomial, and multiple feature classification
│   └── economic_imputation_classification.py # Feature imputation + development status classification
├── output/                                # Subdirectory for generated evaluation figures
├── .gitignore                             # Specifies files ignored by git (e.g., raw PDFs, cache)
├── requirements.txt                       # Package dependencies
└── README.md                              # Main documentation (this file)
```

---

## 📊 Dataset & Features

The dataset (`Life-Expectancy-Data-Updated.csv`) tracks **21 features** across multiple countries over a 16-year span (2000-2015). Key features include:
- **Demographics & Health**: `Life_expectancy`, `Adult_mortality`, `Infant_deaths`, `Under_five_deaths`, `BMI`
- **Immunization & Disease**: `Hepatitis_B`, `Measles`, `Polio`, `Diphtheria`, `Incidents_HIV`
- **Economic & Social Factors**: `GDP_per_capita`, `Population_mln`, `Schooling`, `Alcohol_consumption`
- **Target / Economy Labels**: `Economy_status_Developed`, `Economy_status_Developing` (unified into a binary `economy_status` variable where `Developed = 1`, `Developing = 0`)

---

## 🛠️ Methodologies & Models

### 1. Life Expectancy Prediction (`src/life_expectancy_regression.py`)
- **Single-Feature Regression**: Establishes a baseline using `Adult_mortality` to predict `Life_expectancy`.
- **Multi-Feature Regression**: Dynamically selects variables with high correlation ($|r| \ge 0.39$) to train a multi-variable Linear Regression model.
- **Polynomial Regression**: Explores non-linear curves on `Adult_mortality` by checking degrees 2 through 10 and comparing the resulting Mean Squared Error (MSE) and R² scores.

### 2. Feature Imputation & Prediction (`src/life_expectancy_imputation.py`)
- Simulates a scenario where future feature values (2013-2015) are missing.
- Implements individual cross-feature Linear Regression models: each chosen feature is predicted using a combination of all other selected features.
- Reconstructs the 2013-2015 dataset by inverse-transforming predictions and trains a final regression pipeline to evaluate life expectancy prediction accuracy on imputed data.

### 3. Economy Status Classification (`src/economic_classification.py`)
- Maps development labels to a single binary classifier target: `economy_status`.
- **Single-Feature Classification**: Applies Logistic Regression on `Alcohol_consumption` to classify status.
- **Polynomial Classification**: Trains polynomial Logistic Regression models (degrees 2-10) to map potential non-linear classification boundaries.
- **Multi-Feature Classification**: Automatically queries features with correlation ($|r| \ge 0.20$) to train a multi-variable classification model. 
- **Evaluation Metrics**: Models are assessed via **MSE (Brier Score)**, **Sensitivity (Recall for Developed countries)**, and **Specificity (Recall for Developing countries)**.

### 4. Imputed Feature Classification (`src/economic_imputation_classification.py`)
- Imputes test indicators (2013-2015) via cross-feature regression models.
- Trains a final Logistic Regression classifier on the combined (imputed + historical) feature space to predict the economic class.

---

## 📈 Key Findings

- **Regression**: Integrating multiple correlated features significantly drops the Mean Squared Error (MSE) of Life Expectancy prediction compared to using a single-feature baseline. 
- **Imputation**: Reconstructing missing data via multi-variable regression maintains prediction integrity, yielding competitive MSE metrics against models evaluated on fully observed datasets.
- **Classification**: Multi-feature Logistic Regression achieves excellent classification power, balancing Sensitivity and Specificity far better than single-feature indicators like alcohol consumption alone.

---

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.8+
- Recommended: A virtual environment (`venv`)

### Setup
1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset `Life-Expectancy-Data-Updated.csv` is located in the root directory or inside a `data/` folder.

### Running the Pipelines

You can execute the python scripts using standard terminal command lines. Each script generates summary metrics in the terminal and saves diagnostic plots inside the `output/` directory:

1. **Life Expectancy Regression Model**:
   ```bash
   python src/life_expectancy_regression.py --type all
   ```
   *(Options: `--type single`, `--type multiple`, `--type polynomial`, `--type all`)*

2. **Feature Imputation & Life Expectancy Prediction**:
   ```bash
   python src/life_expectancy_imputation.py --eval-imputation
   ```
   *(Optionally include `--eval-imputation` to print cross-feature prediction errors and generate detailed subplots)*

3. **Development Status Classification**:
   ```bash
   python src/economic_classification.py --type all
   ```
   *(Options: `--type single`, `--type polynomial`, `--type multiple`, `--type all`)*

4. **Economic Status Classification on Imputed Data**:
   ```bash
   python src/economic_imputation_classification.py
   ```
