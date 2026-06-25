import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

def select_features(df, threshold=0.45):
    """Selects features based on absolute correlation with Life Expectancy."""
    numeric_df = df.select_dtypes(include='number')
    correlation = numeric_df.drop(columns='Life_expectancy').corrwith(df['Life_expectancy']).abs()
    selected = correlation[correlation >= threshold].index.tolist()
    return selected

def evaluate_imputation_models(df, selected_features, save_plots=True):
    """
    Evaluates how well each feature can be predicted using the other selected features.
    Saves subplots for actual vs predicted values of each feature.
    """
    print("\n" + "="*50)
    print("EVALUATING INDIVIDUAL FEATURE IMPUTATION MODELS")
    print("="*50)

    # Prepare dataset
    data = df[selected_features + ['Year']]
    train_data = data[data['Year'] <= 2012].drop(columns=['Year'])
    test_data = data[data['Year'] > 2012].drop(columns=['Year'])

    # Scale the features
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

    mse_results = {}
    
    # Split features across two plots to avoid clutter
    midpoint = len(selected_features) // 2 + len(selected_features) % 2
    feature_sets = [selected_features[:midpoint], selected_features[midpoint:]]

    for fig_index, feature_subset in enumerate(feature_sets, start=1):
        nrows = (len(feature_subset) + 2) // 3
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(18, 5 * nrows))
        axes = axes.flatten()

        for i, feature in enumerate(feature_subset):
            # Prepare inputs (all other features) and target (current feature)
            X_train = train_scaled.drop(columns=[feature])
            y_train = train_scaled[feature]
            X_test = test_scaled.drop(columns=[feature])
            y_test = test_scaled[feature]

            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mse_results[feature] = mse

            # Plot on subplot
            ax = axes[i]
            ax.scatter(y_test, y_pred, alpha=0.6, color='#4361ee')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
            ax.set_title(f'{feature}\nMSE: {mse:.4f}', fontsize=12)
            ax.set_xlabel('Actual (scaled)', fontsize=10)
            ax.set_ylabel('Predicted (scaled)', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)

        # Hide any unused axes
        for j in range(len(feature_subset), len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f'Feature Imputation Performance (Figure {fig_index})', fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_plots:
            plot_path = os.path.join("output", f"imputation_features_fig{fig_index}.png")
            plt.savefig(plot_path, dpi=300)
            print(f"Subplot figure {fig_index} saved to: {plot_path}")
        plt.close()

    print("\nMean Squared Error for each predicted feature:")
    for feature, mse in mse_results.items():
        print(f" - {feature:<30}: {mse:.4f}")

    return mse_results

def run_imputed_regression(df, correlation_threshold=0.45, save_plots=True):
    """
    Imputes feature values for 2013-2015 using Linear Regression, and then trains
    a final Linear Regression model on the combined data to predict Life Expectancy.
    """
    print("\n" + "="*50)
    print("RUNNING REGRESSION WITH IMPUTED FUTURE FEATURES")
    print("="*50)

    selected_features = select_features(df, correlation_threshold)
    print(f"Selected Features for Imputation & Regression: {selected_features}")

    # Prepare features data
    data = df[selected_features + ['Year']].copy()
    train_data = data[data['Year'] <= 2012].drop(columns='Year')
    test_data = data[data['Year'] > 2012].drop(columns='Year')

    # Fit scaling scaler on train features
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

    # Impute test data (2013-2015) features
    full_data = df[selected_features + ['Year']].copy()
    combined_data = full_data.copy()

    for feature in selected_features:
        # Train a model on scaling-train data using all features EXCEPT the current one to predict it
        X_train = train_scaled.drop(columns=[feature])
        y_train = train_scaled[feature]
        X_test = test_scaled.drop(columns=[feature])

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Reconstruct scaling-scaled row and inverse scale to get values in original range
        test_scaled_copy = test_scaled.copy()
        test_scaled_copy[feature] = y_pred
        y_pred_original = scaler.inverse_transform(test_scaled_copy)[:, selected_features.index(feature)]

        test_indices = full_data[full_data['Year'] > 2012].index
        combined_data[feature] = combined_data[feature].astype(float)
        combined_data.loc[test_indices, feature] = y_pred_original

    # Append original Life Expectancy target column
    combined_data['Life_expectancy'] = df['Life_expectancy']

    # Final split
    train_combined = combined_data[combined_data['Year'] <= 2012]
    test_combined = combined_data[combined_data['Year'] > 2012]

    X_train_final = train_combined.drop(columns=['Life_expectancy', 'Year'])
    y_train_final = train_combined['Life_expectancy']
    X_test_final = test_combined.drop(columns=['Life_expectancy', 'Year'])
    y_test_final = test_combined['Life_expectancy']

    # Scale final features
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_final)
    X_test_scaled = scaler_final.transform(X_test_final)

    # Train final model to predict Life Expectancy
    model_final = LinearRegression()
    model_final.fit(X_train_scaled, y_train_final)
    y_pred_final = model_final.predict(X_test_scaled)

    mse = mean_squared_error(y_test_final, y_pred_final)
    print(f"\nFinal Life Expectancy Prediction MSE (Using Imputed Features): {mse:.4f}")

    # Plot actual vs predicted life expectancy
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_final, y_pred_final, alpha=0.6, color='#7209b7')
    plt.plot([y_test_final.min(), y_test_final.max()], [y_test_final.min(), y_test_final.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Life Expectancy', fontsize=12)
    plt.ylabel('Predicted Life Expectancy', fontsize=12)
    plt.title('Actual vs Predicted Life Expectancy\n(Using Imputed Future Feature Set)', fontsize=14, pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    if save_plots:
        plot_path = os.path.join("output", "regression_imputed.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Final prediction plot saved to: {plot_path}")
    plt.close()

    return {"mse": mse, "model": model_final}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Life Expectancy Imputation & Regression Pipeline")
    parser.add_argument("--data", type=str, default="Life-Expectancy-Data-Updated.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Correlation threshold for feature selection (default: 0.45)")
    parser.add_argument("--eval-imputation", action="store_true",
                        help="Run evaluation on individual feature imputation models")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot saving")
    args = parser.parse_args()

    # Load data
    try:
        data_df = load_data(args.data)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    selected = select_features(data_df, args.threshold)
    save = not args.no_plot

    if args.eval_imputation:
        evaluate_imputation_models(data_df, selected, save_plots=save)
    
    run_imputed_regression(data_df, args.threshold, save_plots=save)
