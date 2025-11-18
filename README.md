# 🏠 Paris Housing Price Prediction - End-to-End ML Project

A comprehensive machine learning project that predicts housing prices in Paris using multi-linear regression. This project includes data exploration, preprocessing, model training with cross-validation, and an interactive Streamlit demo application.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Linear Regression Assumptions](#linear-regression-assumptions)
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project demonstrates a complete machine learning pipeline:

1. **Data Exploration & Cleaning**: Comprehensive EDA and data preprocessing
2. **Assumption Testing**: Validation of linear regression assumptions
3. **Model Training**: Multi-linear regression with 80/20 train-test split
4. **Cross-Validation**: 10-fold cross-validation for robust evaluation
5. **Interactive Demo**: Streamlit app for real-time price predictions

## ✨ Features

- 📊 Comprehensive data exploration and visualization
- 🧹 Data cleaning and preprocessing pipeline
- ✅ Linear regression assumption testing
- 🤖 Multi-linear regression model training
- 📈 Model evaluation with multiple metrics (RMSE, MAE, R²)
- 🔄 10-fold cross-validation
- 🎨 Interactive Streamlit web application
- 📉 Feature importance analysis
- 📸 Visualization of results and residuals

## 📁 Project Structure

```
linear-regression/
│
├── data/
│   ├── ParisHousing.csv              # Original dataset
│   └── ParisHousing_cleaned.csv      # Cleaned dataset (generated)
│
├── src/
│   ├── explore_and_preprocess.py     # Data exploration & preprocessing
│   └── train_model.py                # Model training script
│
├── models/
│   ├── linear_regression_model.pkl   # Trained model (generated)
│   ├── scaler.pkl                    # Feature scaler (generated)
│   ├── feature_names.json            # Feature names (generated)
│   ├── feature_info.json             # Feature metadata (generated)
│   ├── feature_importance.csv        # Feature importance (generated)
│   ├── model_metrics.json            # Model performance metrics (generated)
│   ├── exploration_plots.png         # EDA visualizations (generated)
│   └── model_results.png             # Model evaluation plots (generated)
│
├── app.py                            # Streamlit demo application
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore file
└── README.md                         # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd linear-regression
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

Run the complete pipeline with a single command:

```bash
python run_pipeline.py
```

This will execute:
1. Data exploration and preprocessing
2. Model training with cross-validation
3. Generate all visualizations and model artifacts

## 📖 Usage

### Step 1: Data Exploration and Preprocessing

Run the exploration and preprocessing script:

```bash
python src/explore_and_preprocess.py
```

This script will:
- Load and explore the dataset
- Check for missing values and duplicates
- Test linear regression assumptions
- Clean the data
- Generate visualizations
- Save cleaned data and feature information

**Output:**
- `data/ParisHousing_cleaned.csv` - Cleaned dataset
- `models/feature_info.json` - Feature metadata
- `models/exploration_plots.png` - EDA visualizations

### Step 2: Model Training

Train the linear regression model:

```bash
python src/train_model.py
```

This script will:
- Load the cleaned data
- Split data into 80% train and 20% test sets
- Scale features using StandardScaler
- Train a multi-linear regression model
- Evaluate model performance
- Perform 10-fold cross-validation
- Generate evaluation plots
- Save the trained model and artifacts

**Output:**
- `models/linear_regression_model.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/feature_names.json` - Feature names
- `models/model_metrics.json` - Performance metrics
- `models/feature_importance.csv` - Feature importance
- `models/model_results.png` - Evaluation visualizations

### Step 3: Run Streamlit App

Launch the interactive demo application:

```bash
streamlit run app.py
```

The app will open in your default web browser. Use the sidebar to:
- Adjust property features (square meters, rooms, floors, etc.)
- Toggle amenities (pool, yard, garage, etc.)
- Get real-time price predictions
- View model performance metrics
- Explore feature importance

## 📊 Model Performance

The model is evaluated using multiple metrics:

- **R² Score**: Measures the proportion of variance explained
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Cross-Validation**: 10-fold CV for robust evaluation

Example performance metrics:
- Test R²: ~0.99+ (excellent fit)
- Test RMSE: Varies based on data
- Cross-Validation R²: Consistent across folds

## ✅ Linear Regression Assumptions

The project tests and validates key linear regression assumptions:

1. **Linearity**: Relationships between features and target are linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No Multicollinearity**: Features are not highly correlated
6. **No Perfect Multicollinearity**: No constant or redundant features

The preprocessing script includes checks for these assumptions and provides recommendations.

## 🎨 Streamlit App Features

The interactive web application includes:

- **Feature Input**: Sliders and checkboxes for all property features
- **Real-time Prediction**: Instant price predictions as you adjust features
- **Model Metrics**: Display of model performance (R², RMSE)
- **Feature Importance**: Visualization of top 10 most important features
- **Quick Examples**: Pre-configured examples for different property types
- **Responsive Design**: Clean, modern UI with intuitive controls

## 🔧 Technical Details

### Dataset

The Paris Housing dataset contains:
- **Features**: 16 features including square meters, number of rooms, amenities, location, etc.
- **Target**: Housing price (continuous variable)
- **Size**: ~10,000 observations

### Model Architecture

- **Algorithm**: Multi-Linear Regression (Ordinary Least Squares)
- **Preprocessing**: StandardScaler for feature normalization
- **Validation**: 80/20 train-test split + 10-fold cross-validation
- **Evaluation**: Multiple metrics (R², RMSE, MAE)

### Key Libraries

- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning
- `matplotlib` & `seaborn`: Visualization
- `streamlit`: Web application
- `scipy`: Statistical tests

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset: Paris Housing dataset
- Libraries: scikit-learn, pandas, streamlit, and the open-source community

## 🚀 Pushing to GitHub

To push this project to GitHub:

1. **Create a new repository on GitHub** (don't initialize with README)

2. **Add the remote and push**:
   ```bash
   git add .
   git commit -m "Initial commit: Paris Housing Price Prediction ML Project"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

3. **Replace** `YOUR_USERNAME` and `YOUR_REPO_NAME` with your GitHub username and repository name

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Predicting! 🎉**

