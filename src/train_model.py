"""
Model Training Script
Implements 80/20 train-test split and 10-fold cross-validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_cleaned_data(file_path='data/ParisHousing_cleaned.csv'):
    """Load the cleaned dataset"""
    print("=" * 60)
    print("LOADING CLEANED DATA")
    print("=" * 60)
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df

def prepare_features(X, y, feature_info_path='models/feature_info.json'):
    """Prepare features for modeling"""
    print("\n" + "=" * 60)
    print("PREPARING FEATURES")
    print("=" * 60)
    
    # Load feature info
    with open(feature_info_path, 'r') as f:
        feature_info = json.load(f)
    
    # All features are numeric in this dataset, so no encoding needed
    # But we'll standardize for better model performance
    X_processed = X.copy()
    
    print(f"Features prepared: {X_processed.shape[1]} features")
    print(f"Sample size: {X_processed.shape[0]} observations")
    
    return X_processed, feature_info

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets (80/20)"""
    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT (80/20)")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    print("\n" + "=" * 60)
    print("SCALING FEATURES")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train, scale_features_flag=True):
    """Train the linear regression model"""
    print("\n" + "=" * 60)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("=" * 60)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Model trained successfully!")
    print(f"Coefficients: {len(model.coef_)} features")
    print(f"Intercept: {model.intercept_:.2f}")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Evaluate model performance"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTraining Set Metrics:")
    print(f"  RMSE: {train_rmse:,.2f}")
    print(f"  MAE:  {train_mae:,.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print("\nTest Set Metrics:")
    print(f"  RMSE: {test_rmse:,.2f}")
    print(f"  MAE:  {test_mae:,.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # Feature importance (coefficients)
    print("\nTop 10 Most Important Features (by absolute coefficient):")
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'feature_importance': feature_importance
    }

def cross_validate_model(model, X_train, y_train, cv=10):
    """Perform 10-fold cross-validation"""
    print("\n" + "=" * 60)
    print(f"{cv}-FOLD CROSS-VALIDATION")
    print("=" * 60)
    
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores_rmse = -cross_val_score(
        model, X_train, y_train, 
        cv=kfold, scoring='neg_root_mean_squared_error'
    )
    cv_scores_r2 = cross_val_score(
        model, X_train, y_train, 
        cv=kfold, scoring='r2'
    )
    cv_scores_mae = -cross_val_score(
        model, X_train, y_train, 
        cv=kfold, scoring='neg_mean_absolute_error'
    )
    
    print(f"\nCross-Validation Results ({cv} folds):")
    print(f"\nRMSE:")
    print(f"  Mean: {cv_scores_rmse.mean():,.2f}")
    print(f"  Std:  {cv_scores_rmse.std():,.2f}")
    print(f"  Min:  {cv_scores_rmse.min():,.2f}")
    print(f"  Max:  {cv_scores_rmse.max():,.2f}")
    
    print(f"\nMAE:")
    print(f"  Mean: {cv_scores_mae.mean():,.2f}")
    print(f"  Std:  {cv_scores_mae.std():,.2f}")
    
    print(f"\nR²:")
    print(f"  Mean: {cv_scores_r2.mean():.4f}")
    print(f"  Std:  {cv_scores_r2.std():.4f}")
    print(f"  Min:  {cv_scores_r2.min():.4f}")
    print(f"  Max:  {cv_scores_r2.max():.4f}")
    
    return {
        'cv_rmse_mean': cv_scores_rmse.mean(),
        'cv_rmse_std': cv_scores_rmse.std(),
        'cv_mae_mean': cv_scores_mae.mean(),
        'cv_mae_std': cv_scores_mae.std(),
        'cv_r2_mean': cv_scores_r2.mean(),
        'cv_r2_std': cv_scores_r2.std(),
        'cv_scores_rmse': cv_scores_rmse,
        'cv_scores_r2': cv_scores_r2
    }

def plot_results(y_train, y_test, y_train_pred, y_test_pred, cv_results):
    """Create visualization plots"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Actual vs Predicted - Training
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5)
    axes[0, 0].plot([y_train.min(), y_train.max()], 
                    [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price')
    axes[0, 0].set_ylabel('Predicted Price')
    axes[0, 0].set_title('Training Set: Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted - Test
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Price')
    axes[0, 1].set_ylabel('Predicted Price')
    axes[0, 1].set_title('Test Set: Actual vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals - Training
    residuals_train = y_train - y_train_pred
    axes[0, 2].scatter(y_train_pred, residuals_train, alpha=0.5)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('Predicted Price')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].set_title('Training Set: Residual Plot')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Residuals - Test
    residuals_test = y_test - y_test_pred
    axes[1, 0].scatter(y_test_pred, residuals_test, alpha=0.5, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Price')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Test Set: Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Distribution of Residuals
    axes[1, 1].hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Test Set Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Cross-Validation Scores
    axes[1, 2].boxplot([cv_results['cv_scores_r2']], labels=['R² Score'])
    axes[1, 2].set_ylabel('R² Score')
    axes[1, 2].set_title('10-Fold Cross-Validation R² Scores')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/model_results.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to models/model_results.png")
    plt.close()

def save_model(model, scaler, feature_names, metrics, cv_results):
    """Save the trained model and related artifacts"""
    print("\n" + "=" * 60)
    print("SAVING MODEL AND ARTIFACTS")
    print("=" * 60)
    
    # Save model
    joblib.dump(model, 'models/linear_regression_model.pkl')
    print("Model saved to: models/linear_regression_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to: models/scaler.pkl")
    
    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names.tolist(), f, indent=2)
    print("Feature names saved to: models/feature_names.json")
    
    # Save metrics
    metrics_to_save = {
        'train_rmse': float(metrics['train_rmse']),
        'test_rmse': float(metrics['test_rmse']),
        'train_mae': float(metrics['train_mae']),
        'test_mae': float(metrics['test_mae']),
        'train_r2': float(metrics['train_r2']),
        'test_r2': float(metrics['test_r2']),
        'cv_rmse_mean': float(cv_results['cv_rmse_mean']),
        'cv_rmse_std': float(cv_results['cv_rmse_std']),
        'cv_r2_mean': float(cv_results['cv_r2_mean']),
        'cv_r2_std': float(cv_results['cv_r2_std'])
    }
    
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print("Model metrics saved to: models/model_metrics.json")
    
    # Save feature importance
    metrics['feature_importance'].to_csv('models/feature_importance.csv', index=False)
    print("Feature importance saved to: models/feature_importance.csv")

def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("PARIS HOUSING - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    df = load_cleaned_data()
    
    # Prepare features
    X = df.drop(columns=['price'])
    y = df['price']
    X_processed, feature_info = prepare_features(X, y)
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split_data(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    metrics = evaluate_model(
        model, X_train_scaled, X_test_scaled, 
        y_train, y_test, X_train.columns
    )
    
    # Cross-validation
    cv_results = cross_validate_model(model, X_train_scaled, y_train, cv=10)
    
    # Plot results
    plot_results(
        y_train, y_test, 
        metrics['y_train_pred'], metrics['y_test_pred'],
        cv_results
    )
    
    # Save model and artifacts
    save_model(
        model, scaler, X_train.columns, 
        metrics, cv_results
    )
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModel Performance Summary:")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:,.2f}")
    print(f"  CV R² (mean ± std): {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
    
    return model, scaler, metrics, cv_results

if __name__ == "__main__":
    main()

