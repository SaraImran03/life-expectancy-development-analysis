import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
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
        # Combine economy columns into a single binary column
        df_copy['economy_status'] = df_copy['Economy_status_Developed'].apply(lambda x: 1 if x == 1 else 0)
        # Drop old economy columns to avoid data leakage
        df_copy.drop(columns=['Economy_status_Developed', 'Economy_status_Developing'], inplace=True, errors='ignore')
    return df_copy.dropna()

def select_features(df, threshold=0.2):
    """Selects numeric features based on absolute correlation with economy_status."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Compute absolute correlation with economy status, excluding target itself
    correlations = df[numeric_cols].corr()['economy_status'].drop('economy_status', errors='ignore')
    selected_features = correlations[abs(correlations) >= threshold].index.tolist()
    
    # Exclude 'Year' as a feature
    if 'Year' in selected_features:
        selected_features.remove('Year')
        
    return selected_features, correlations

def evaluate_imputation_models(train_df, test_df, selected_features, save_plots=True):
    """
    Evaluates individual feature prediction models.
    Saves a consolidated subplot of all feature predictions.
    """
    print("\n" + "="*50)
    print("EVALUATING INDIVIDUAL FEATURE IMPUTATION MODELS (CLASSIFICATION SET)")
    print("="*50)

    num_features = len(selected_features)
    nrows = (num_features + 2) // 3
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(18, 5 * nrows))
    axes = axes.flatten()

    for idx, feature in enumerate(selected_features):
        input_features = [f for f in selected_features if f != feature]

        X_train = train_df[input_features]
        y_train = train_df[feature]
        X_test = test_df[input_features]
        y_test = test_df[feature]

        # Feature scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # Fit model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)

        # Predict
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Compute MSE
        mse = mean_squared_error(y_test, y_pred)
        print(f"Feature: {feature:<25} | Imputation MSE: {mse:.4f}")

        # Subplot
        ax = axes[idx]
        ax.scatter(y_test, y_pred, alpha=0.6, color='#2a9d8f')
        ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=2)
        ax.set_title(f"{feature}\nMSE: {mse:.4f}", fontsize=12)
        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)

    # Hide unused subplots
    for idx in range(num_features, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("Feature Imputation: Actual vs Predicted Features (Test Set)", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plots:
        plot_path = os.path.join("output", "imputation_classification_features.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Feature imputation plots saved to: {plot_path}")
    plt.close()

def run_imputed_classification(df, correlation_threshold=0.2, save_plots=True):
    """
    Imputes economic features for test set using Linear Regression and then trains
    Logistic Regression on combined data to classify Economy Status.
    """
    print("\n" + "="*50)
    print("RUNNING CLASSIFICATION ON IMPUTED FEATURE DATA")
    print("="*50)

    selected_features, correlations = select_features(df, correlation_threshold)
    print("Selected features (correlation >= 0.2):")
    print(selected_features)

    # Split by year
    train_df = df[df['Year'] <= 2012].copy()
    test_df = df[df['Year'] > 2012].copy()

    # Perform evaluation/visualization of imputation if requested
    evaluate_imputation_models(train_df, test_df, selected_features, save_plots=save_plots)

    # Impute test features
    predicted_test_features = pd.DataFrame(index=test_df.index)
    for feature in selected_features:
        input_features = [f for f in selected_features if f != feature]

        X_train = train_df[input_features]
        y_train = train_df[feature]
        X_test = test_df[input_features]

        # Feature scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        model = LinearRegression()
        model.fit(X_train_scaled, y_train_scaled)

        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        predicted_test_features[feature] = y_pred

    # Combine original (2000-2012) and predicted (2013-2015) features
    combined_features = pd.concat([
        train_df[selected_features],
        predicted_test_features
    ])

    combined_target = pd.concat([
        train_df['economy_status'],
        test_df['economy_status']
    ])

    # Final split
    X_train_final = combined_features.loc[train_df.index]
    X_test_final = combined_features.loc[test_df.index]
    y_train_final = combined_target.loc[train_df.index]
    y_test_final = combined_target.loc[test_df.index]

    # Feature scaling
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_final)
    X_test_scaled = scaler_final.transform(X_test_final)

    # Logistic Regression model training
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train_final)
    y_pred_final = log_reg.predict(X_test_scaled)

    # Evaluation
    mse_final = mean_squared_error(y_test_final, y_pred_final)
    cm = confusion_matrix(y_test_final, y_pred_final)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    print("\nFinal Logistic Regression Results on Imputed Dataset:")
    print(f"MSE (economy_status prediction): {mse_final:.4f}")
    print(f"Sensitivity (Developed): {sensitivity:.3f}")
    print(f"Specificity (Developing): {specificity:.3f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=["Developing", "Developed"], yticklabels=["Developing", "Developed"])
    plt.title("Confusion Matrix: Logistic Regression (Imputed Features)", fontsize=12, pad=15)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("Actual Class", fontsize=10)
    plt.tight_layout()

    if save_plots:
        plot_path = os.path.join("output", "classification_imputed.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Final confusion matrix saved to: {plot_path}")
    plt.close()

    return {"mse": mse_final, "sensitivity": sensitivity, "specificity": specificity}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Economic Status Imputation & Classification Pipeline")
    parser.add_argument("--data", type=str, default="Life-Expectancy-Data-Updated.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Correlation threshold for feature selection (default: 0.2)")
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

    run_imputed_classification(processed_df, correlation_threshold=args.threshold, save_plots=save)
