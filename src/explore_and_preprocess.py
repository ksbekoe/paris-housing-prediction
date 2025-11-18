"""
Data Exploration, Cleaning, and Preprocessing Script
Tests linear regression assumptions and prepares data for modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(file_path):
    """Load the dataset"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    return df

def explore_data(df):
    """Comprehensive data exploration"""
    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    # Basic info
    print("\n1. Dataset Info:")
    print(df.info())
    
    # Statistical summary
    print("\n2. Statistical Summary:")
    print(df.describe())
    
    # Check for missing values
    print("\n3. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    # Check for duplicates
    print(f"\n4. Duplicate Rows: {df.duplicated().sum()}")
    
    # Data types
    print("\n5. Data Types:")
    print(df.dtypes)
    
    # Check for outliers using IQR method
    print("\n6. Outlier Detection (IQR Method):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_summary[col] = len(outliers)
        if len(outliers) > 0:
            print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
    
    return df

def visualize_data(df, save_plots=True):
    """Create visualizations for data exploration"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Distribution of target variable
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    df['price'].hist(bins=50, edgecolor='black')
    plt.title('Distribution of Price (Target Variable)')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    # Correlation heatmap
    plt.subplot(2, 3, 2)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap')
    
    # Price vs key features
    plt.subplot(2, 3, 3)
    plt.scatter(df['squareMeters'], df['price'], alpha=0.5)
    plt.xlabel('Square Meters')
    plt.ylabel('Price')
    plt.title('Price vs Square Meters')
    
    plt.subplot(2, 3, 4)
    plt.scatter(df['numberOfRooms'], df['price'], alpha=0.5)
    plt.xlabel('Number of Rooms')
    plt.ylabel('Price')
    plt.title('Price vs Number of Rooms')
    
    # Box plot for categorical features
    plt.subplot(2, 3, 5)
    df.boxplot(column='price', by='hasPool', ax=plt.gca())
    plt.title('Price Distribution by Pool')
    plt.suptitle('')
    
    plt.subplot(2, 3, 6)
    df.boxplot(column='price', by='hasYard', ax=plt.gca())
    plt.title('Price Distribution by Yard')
    plt.suptitle('')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('models/exploration_plots.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved to models/exploration_plots.png")
    plt.close()

def test_linear_regression_assumptions(df, target_col='price'):
    """Test linear regression assumptions"""
    print("\n" + "=" * 60)
    print("TESTING LINEAR REGRESSION ASSUMPTIONS")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Get numeric features only for assumption testing
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n1. LINEARITY ASSUMPTION")
    print("   Checking linear relationships between features and target...")
    # We'll check this visually and with correlation
    correlations = {}
    for feature in numeric_features[:10]:  # Check first 10 numeric features
        corr = df[feature].corr(y)
        correlations[feature] = corr
        if abs(corr) > 0.3:
            print(f"   ✓ {feature}: correlation = {corr:.3f} (moderate to strong)")
        else:
            print(f"   ⚠ {feature}: correlation = {corr:.3f} (weak)")
    
    print("\n2. INDEPENDENCE ASSUMPTION")
    print("   Assuming observations are independent (no time series structure)")
    print("   ✓ Data appears to be cross-sectional")
    
    print("\n3. HOMOSCEDASTICITY (Constant Variance)")
    print("   Will be tested after model fitting")
    print("   ⚠ To be verified with residual plots")
    
    print("\n4. NORMALITY OF ERRORS")
    print("   Will be tested after model fitting")
    print("   ⚠ To be verified with Q-Q plots and statistical tests")
    
    print("\n5. MULTICOLLINEARITY CHECK")
    print("   Checking for high correlations between features...")
    X_numeric = X[numeric_features]
    corr_matrix = X_numeric.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_pairs = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            if upper_triangle.loc[idx, col] > 0.8:
                high_corr_pairs.append((idx, col, upper_triangle.loc[idx, col]))
    
    if high_corr_pairs:
        print("   ⚠ High multicollinearity detected:")
        for pair in high_corr_pairs[:5]:  # Show first 5
            print(f"      {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
    else:
        print("   ✓ No severe multicollinearity detected")
    
    print("\n6. NO PERFECT MULTICOLLINEARITY")
    # Check for constant or near-constant features
    constant_features = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_features.append(col)
        elif X[col].nunique() == 2 and X[col].dtype in ['int64', 'float64']:
            # Check if it's essentially constant (99% same value)
            value_counts = X[col].value_counts()
            if value_counts.iloc[0] / len(X) > 0.99:
                constant_features.append(col)
    
    if constant_features:
        print(f"   ⚠ Constant/near-constant features found: {constant_features}")
    else:
        print("   ✓ No perfect multicollinearity detected")
    
    return correlations, high_corr_pairs

def clean_data(df):
    """Clean the dataset"""
    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)
    
    df_clean = df.copy()
    initial_shape = df_clean.shape
    
    # Remove duplicates
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    
    # Handle missing values (if any)
    missing = df_clean.isnull().sum().sum()
    if missing > 0:
        # For numeric columns, fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        print(f"Handled {missing} missing values")
    else:
        print("No missing values to handle")
    
    # Remove outliers for target variable (optional - we'll keep them for now)
    # Outliers might be legitimate high-value properties
    
    final_shape = df_clean.shape
    print(f"\nData shape: {initial_shape} -> {final_shape}")
    
    return df_clean

def prepare_data_for_modeling(df, target_col='price'):
    """Prepare data for machine learning"""
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR MODELING")
    print("=" * 60)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    binary_features = [col for col in X.columns if X[col].nunique() == 2]
    
    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Binary features ({len(binary_features)}): {binary_features}")
    
    return X, y, numeric_features, categorical_features, binary_features

def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("PARIS HOUSING DATA - EXPLORATION & PREPROCESSING")
    print("=" * 60)
    
    # Load data
    df = load_data('data/ParisHousing.csv')
    
    # Explore data
    df = explore_data(df)
    
    # Visualize data
    visualize_data(df)
    
    # Test assumptions
    correlations, multicollinearity = test_linear_regression_assumptions(df)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Prepare for modeling
    X, y, numeric_features, categorical_features, binary_features = prepare_data_for_modeling(df_clean)
    
    # Save cleaned data
    df_clean.to_csv('data/ParisHousing_cleaned.csv', index=False)
    print("\n" + "=" * 60)
    print("CLEANED DATA SAVED TO: data/ParisHousing_cleaned.csv")
    print("=" * 60)
    
    # Save feature information
    feature_info = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'binary_features': binary_features,
        'target': 'price'
    }
    
    import json
    with open('models/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("\nFeature information saved to: models/feature_info.json")
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    
    return df_clean, X, y, feature_info

if __name__ == "__main__":
    main()






