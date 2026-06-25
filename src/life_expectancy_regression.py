import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

def load_data(filepath="Life-Expectancy-Data-Updated.csv"):
    """Loads the dataset from the specified file path or fallback locations."""
    fallbacks = [filepath, os.path.join("data", filepath), os.path.join("..", filepath)]
    for path in fallbacks:
        if os.path.exists(path):
            print(f"Loading dataset from: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(f"Dataset '{filepath}' not found in any of the expected locations.")

def run_single_regression(df, save_plots=True):
    """Performs single-feature linear regression using Adult Mortality to predict Life Expectancy."""
    print("\n" + "="*50)
    print("RUNNING SINGLE-FEATURE LINEAR REGRESSION")
    print("="*50)

    # Select relevant columns
    data = df[['Adult_mortality', 'Life_expectancy', 'Year']]

    # Split data by Year (<= 2012 for train, > 2012 for test)
    train_data = data[data['Year'] <= 2012]
    test_data = data[data['Year'] > 2012]

    # Separate features and target
    X_train = train_data[['Adult_mortality']]
    y_train = train_data['Life_expectancy']
    X_test = test_data[['Adult_mortality']]
    y_test = test_data['Life_expectancy']

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Feature Used: Adult_mortality")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Model Coefficient: {model.coef_[0]:.4f}")
    print(f"Model Intercept: {model.intercept_:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='#7b2cbf', alpha=0.6, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#e63946', linestyle='--', linewidth=2, label='Perfect Prediction Reference')
    plt.title('Actual vs Predicted Life Expectancy\n(Single-Feature Linear Regression: Adult Mortality)', fontsize=14, pad=15)
    plt.xlabel('Actual Life Expectancy', fontsize=12)
    plt.ylabel('Predicted Life Expectancy', fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    if save_plots:
        plot_path = os.path.join("output", "regression_single.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to: {plot_path}")
    plt.close()

    return {"mse": mse, "r2": r2}

def run_multiple_regression(df, correlation_threshold=0.39, save_plots=True):
    """Performs multi-feature linear regression using features selected by correlation threshold."""
    print("\n" + "="*50)
    print("RUNNING MULTI-FEATURE LINEAR REGRESSION")
    print("="*50)

    # Calculate absolute correlation with life expectancy
    numeric_df = df.select_dtypes(include='number')
    correlation = numeric_df.drop(columns='Life_expectancy').corrwith(df['Life_expectancy']).abs()

    # Select features meeting the threshold
    selected_features = correlation[correlation >= correlation_threshold].index.tolist()
    print(f"Selected Features (Correlation >= {correlation_threshold}):")
    for feat in selected_features:
        print(f" - {feat}: r = {df['Life_expectancy'].corr(df[feat]):.4f}")

    # Include Target and Splitting columns
    data = df[selected_features + ['Life_expectancy', 'Year']]

    # Split data by Year
    train_data = data[data['Year'] <= 2012]
    test_data = data[data['Year'] > 2012]

    # Separate features and target
    X_train = train_data.drop(columns=['Life_expectancy', 'Year'])
    y_train = train_data['Life_expectancy']
    X_test = test_data.drop(columns=['Life_expectancy', 'Year'])
    y_test = test_data['Life_expectancy']

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Display Coefficients
    coefficients = pd.Series(model.coef_, index=X_train.columns)
    print("\nFeature Coefficients (sorted by impact):")
    print(coefficients.sort_values(key=abs, ascending=False).to_string())

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='#3f37c9', alpha=0.6, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#f72585', linestyle='--', linewidth=2, label='Perfect Prediction Reference')
    plt.xlabel('Actual Life Expectancy', fontsize=12)
    plt.ylabel('Predicted Life Expectancy', fontsize=12)
    plt.title('Actual vs Predicted Life Expectancy\n(Multi-Feature Linear Regression)', fontsize=14, pad=15)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    if save_plots:
        plot_path = os.path.join("output", "regression_multiple.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to: {plot_path}")
    plt.close()

    return {"mse": mse, "r2": r2, "selected_features": selected_features}

def run_polynomial_regression(df, min_degree=2, max_degree=10, save_plots=True):
    """Performs polynomial regression on Adult Mortality to predict Life Expectancy for degrees 2 to 10."""
    print("\n" + "="*50)
    print("RUNNING POLYNOMIAL REGRESSION ON ADULT MORTALITY")
    print("="*50)

    # Focus on Adult Mortality and Life Expectancy
    data = df[['Adult_mortality', 'Life_expectancy', 'Year']]
    train = data[data['Year'] <= 2012]
    test = data[data['Year'] > 2012]

    # Split into features and target
    X_train = train[['Adult_mortality']]
    y_train = train['Life_expectancy']
    X_test = test[['Adult_mortality']]
    y_test = test['Life_expectancy']

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Set up subplots: 9 plots (degrees 2 to 10)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 15))
    axes = axes.flatten()

    print("Polynomial Regression MSE (Mean Squared Error):")
    print("-" * 50)

    results = {}
    for i, degree in enumerate(range(min_degree, max_degree + 1)):
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)

        # Compute MSE
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[degree] = {"mse": mse, "r2": r2}
        print(f"Degree {degree:2d}: MSE = {mse:.4f} | R² = {r2:.4f}")

        # Scatter plot on subplot
        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.5, color='#4895ef', label='Predicted')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Reference')
        ax.set_title(f'Degree {degree} (MSE: {mse:.2f})', fontsize=12)
        ax.set_xlabel('Actual', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle('Polynomial Regression on Adult Mortality (Degrees 2–10)', fontsize=18, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_plots:
        plot_path = os.path.join("output", "regression_polynomial.png")
        plt.savefig(plot_path, dpi=300)
        print(f"\nAll polynomial plots saved to: {plot_path}")
    plt.close()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Life Expectancy Regression Pipeline")
    parser.add_argument("--type", type=str, choices=["single", "multiple", "polynomial", "all"], default="all",
                        help="Type of regression to run: 'single', 'multiple', 'polynomial', or 'all' (default: 'all')")
    parser.add_argument("--data", type=str, default="Life-Expectancy-Data-Updated.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    args = parser.parse_args()

    # Load data
    try:
        data_df = load_data(args.data)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    save = not args.no_plot

    if args.type in ["single", "all"]:
        run_single_regression(data_df, save_plots=save)
    if args.type in ["multiple", "all"]:
        run_multiple_regression(data_df, save_plots=save)
    if args.type in ["polynomial", "all"]:
        run_polynomial_regression(data_df, save_plots=save)
