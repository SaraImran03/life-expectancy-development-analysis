import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, confusion_matrix

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

def load_data(filepath="Life-Expectancy-Data-Updated.csv"):
    """Loads the dataset from the specified file path or fallback locations."""
    fallbacks = [filepath, os.path.join("data", filepath), os.path.join("..", filepath)]
    for path in fallbacks:
        if os.path.exists(path):
            print(f"Loading dataset from: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(f"Dataset '{filepath}' not found in any of the expected locations.")

def prepare_target(df):
    """Processes the dataset to create a single binary economy status column."""
    df_copy = df.copy()
    if 'economy_status' not in df_copy.columns:
        # Map developed (1) and developing (0)
        df_copy['economy_status'] = df_copy['Economy_status_Developed'].apply(lambda x: 1 if x == 1 else 0)
    return df_copy

def calculate_classification_metrics(y_true, y_pred):
    """Calculates MSE, Sensitivity, and Specificity."""
    mse = mean_squared_error(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    
    return mse, sensitivity, specificity, cm

def run_single_classification(df, save_plots=True):
    """Runs a single-feature Logistic Regression using Alcohol consumption to classify economy status."""
    print("\n" + "="*50)
    print("RUNNING SINGLE-FEATURE LOGISTIC REGRESSION CLASSIFICATION")
    print("="*50)

    # Filter required columns and drop missing values
    data = df[['Alcohol_consumption', 'economy_status', 'Year']].dropna()

    # Train/test split by Year (<= 2012 for train, > 2012 for test)
    train_data = data[data['Year'] <= 2012]
    test_data = data[data['Year'] > 2012]

    X_train = train_data[['Alcohol_consumption']]
    y_train = train_data['economy_status']
    X_test = test_data[['Alcohol_consumption']]
    y_test = test_data['economy_status']

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    mse, sensitivity, specificity, cm = calculate_classification_metrics(y_test, y_pred)

    print(f"Feature Used: Alcohol_consumption")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Sensitivity (Recall for Developed): {sensitivity:.4f}")
    print(f"Specificity (Recall for Developing): {specificity:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Developing", "Developed"], yticklabels=["Developing", "Developed"])
    plt.title("Confusion Matrix: Single-Feature Logistic Regression", fontsize=12, pad=15)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("Actual Class", fontsize=10)
    plt.tight_layout()

    if save_plots:
        plot_path = os.path.join("output", "classification_single.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Confusion matrix plot saved to: {plot_path}")
    plt.close()

    return {"mse": mse, "sensitivity": sensitivity, "specificity": specificity}

def run_polynomial_classification(df, min_degree=2, max_degree=10, save_plots=True):
    """Runs polynomial Logistic Regression on Alcohol consumption for degrees 2 to 10."""
    print("\n" + "="*50)
    print("RUNNING POLYNOMIAL LOGISTIC REGRESSION CLASSIFICATION")
    print("="*50)

    # Filter required columns and drop missing values
    data = df[['Alcohol_consumption', 'economy_status', 'Year']].dropna()

    train_data = data[data['Year'] <= 2012]
    test_data = data[data['Year'] > 2012]

    X_train = train_data[['Alcohol_consumption']]
    y_train = train_data['economy_status']
    X_test = test_data[['Alcohol_consumption']]
    y_test = test_data['economy_status']

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Grid of confusion matrices (3x3 for degrees 2 to 10)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
    axes = axes.flatten()

    print(f"{'Degree':<8} | {'MSE':<8} | {'Sensitivity':<12} | {'Specificity':<12}")
    print("-" * 50)

    results = {}
    for i, degree in enumerate(range(min_degree, max_degree + 1)):
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_poly, y_train)

        # Predict
        y_pred = model.predict(X_test_poly)

        # Metrics
        mse, sensitivity, specificity, cm = calculate_classification_metrics(y_test, y_pred)
        results[degree] = {"mse": mse, "sensitivity": sensitivity, "specificity": specificity}
        
        print(f"{degree:<8d} | {mse:<8.4f} | {sensitivity:<12.4f} | {specificity:<12.4f}")

        # Plot confusion matrix on the subplot
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax,
                    xticklabels=["Developing", "Developed"], yticklabels=["Developing", "Developed"])
        ax.set_title(f"Degree {degree}\nMSE: {mse:.4f}", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

    plt.suptitle("Confusion Matrices: Polynomial Logistic Regression (Degrees 2-10)", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plots:
        plot_path = os.path.join("output", "classification_polynomial.png")
        plt.savefig(plot_path, dpi=300)
        print(f"\nAll confusion matrix subplots saved to: {plot_path}")
    plt.close()

    return results

def run_multiple_classification(df, correlation_threshold=0.2, save_plots=True):
    """Runs multi-feature Logistic Regression using features selected by correlation threshold."""
    print("\n" + "="*50)
    print("RUNNING MULTI-FEATURE LOGISTIC REGRESSION CLASSIFICATION")
    print("="*50)

    # Process features
    numeric_df = df.select_dtypes(include=np.number)
    # Drop original developed/developing columns to avoid leakage
    leakage_cols = ['Economy_status_Developed', 'Economy_status_Developing']
    cols_to_corr = [c for c in numeric_df.columns if c not in leakage_cols and c != 'economy_status']
    
    correlations = numeric_df[cols_to_corr].corrwith(numeric_df['economy_status']).abs()
    correlations = correlations.sort_values(ascending=False)

    selected_features = correlations[correlations >= correlation_threshold].index.tolist()
    print("Selected Features based on Correlation with economy_status:")
    for feat in selected_features:
         print(f" - {feat}: r = {df['economy_status'].corr(df[feat]):.4f}")

    df_selected = df[selected_features + ['economy_status', 'Year']].dropna()

    train_data = df_selected[df_selected['Year'] <= 2012]
    test_data = df_selected[df_selected['Year'] > 2012]

    X_train = train_data.drop(columns=['economy_status', 'Year'])
    y_train = train_data['economy_status']
    X_test = test_data.drop(columns=['economy_status', 'Year'])
    y_test = test_data['economy_status']

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    mse, sensitivity, specificity, cm = calculate_classification_metrics(y_test, y_pred)

    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"Sensitivity (Recall for Developed): {sensitivity:.4f}")
    print(f"Specificity (Recall for Developing): {specificity:.4f}")

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["Developing", "Developed"], yticklabels=["Developing", "Developed"])
    plt.title("Confusion Matrix: Multi-Feature Logistic Regression", fontsize=12, pad=15)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("Actual Class", fontsize=10)
    plt.tight_layout()

    if save_plots:
        plot_path = os.path.join("output", "classification_multiple.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Confusion matrix plot saved to: {plot_path}")
    plt.close()

    return {"mse": mse, "sensitivity": sensitivity, "specificity": specificity, "selected_features": selected_features}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Economic Status Classification Pipeline")
    parser.add_argument("--type", type=str, choices=["single", "polynomial", "multiple", "all"], default="all",
                        help="Type of classification to run: 'single', 'polynomial', 'multiple', or 'all' (default: 'all')")
    parser.add_argument("--data", type=str, default="Life-Expectancy-Data-Updated.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Correlation threshold for multi-feature selection (default: 0.2)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot saving")
    args = parser.parse_args()

    # Load and prepare data
    try:
        raw_df = load_data(args.data)
        processed_df = prepare_target(raw_df)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    save = not args.no_plot

    if args.type in ["single", "all"]:
        run_single_classification(processed_df, save_plots=save)
    if args.type in ["polynomial", "all"]:
        run_polynomial_classification(processed_df, save_plots=save)
    if args.type in ["multiple", "all"]:
        run_multiple_classification(processed_df, correlation_threshold=args.threshold, save_plots=save)
