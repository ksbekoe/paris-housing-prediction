"""
Streamlit Demo App for Paris Housing Price Prediction
Interactive app for users to predict housing prices using key features
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Paris Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('models/linear_regression_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        with open('models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return model, scaler, feature_names, metrics
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run the training script first: {e}")
        st.stop()

def get_feature_inputs(feature_names):
    """Create input widgets for features"""
    inputs = {}
    
    # Load sample data to get ranges
    try:
        df_sample = pd.read_csv('data/ParisHousing_cleaned.csv')
        feature_ranges = {}
        for feature in feature_names:
            feature_ranges[feature] = {
                'min': float(df_sample[feature].min()),
                'max': float(df_sample[feature].max()),
                'mean': float(df_sample[feature].mean()),
                'median': float(df_sample[feature].median())
            }
    except:
        feature_ranges = {}
    
    # Organize features by category
    numeric_features = [f for f in feature_names if f not in 
                       ['hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector', 
                        'hasStorageRoom', 'hasGuestRoom']]
    binary_features = [f for f in feature_names if f in 
                      ['hasYard', 'hasPool', 'isNewBuilt', 'hasStormProtector', 
                       'hasStorageRoom', 'hasGuestRoom']]
    
    # Main features section
    st.sidebar.header("🏠 Property Features")
    
    # Key numeric features
    if 'squareMeters' in numeric_features:
        inputs['squareMeters'] = st.sidebar.slider(
            "Square Meters",
            min_value=int(feature_ranges.get('squareMeters', {}).get('min', 5000)),
            max_value=int(feature_ranges.get('squareMeters', {}).get('max', 100000)),
            value=int(feature_ranges.get('squareMeters', {}).get('mean', 50000)),
            step=100
        )
    
    if 'numberOfRooms' in numeric_features:
        inputs['numberOfRooms'] = st.sidebar.slider(
            "Number of Rooms",
            min_value=int(feature_ranges.get('numberOfRooms', {}).get('min', 1)),
            max_value=int(feature_ranges.get('numberOfRooms', {}).get('max', 100)),
            value=int(feature_ranges.get('numberOfRooms', {}).get('mean', 50)),
            step=1
        )
    
    if 'floors' in numeric_features:
        inputs['floors'] = st.sidebar.slider(
            "Floors",
            min_value=int(feature_ranges.get('floors', {}).get('min', 1)),
            max_value=int(feature_ranges.get('floors', {}).get('max', 100)),
            value=int(feature_ranges.get('floors', {}).get('mean', 50)),
            step=1
        )
    
    # Additional numeric features
    st.sidebar.subheader("Additional Features")
    
    for feature in numeric_features:
        if feature not in ['squareMeters', 'numberOfRooms', 'floors']:
            if feature in feature_ranges:
                min_val = feature_ranges[feature]['min']
                max_val = feature_ranges[feature]['max']
                default_val = feature_ranges[feature]['mean']
            else:
                min_val, max_val, default_val = 0, 10000, 1000
            
            inputs[feature] = st.sidebar.number_input(
                feature.replace('_', ' ').title(),
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=1.0
            )
    
    # Binary features
    st.sidebar.subheader("Property Amenities")
    for feature in binary_features:
        inputs[feature] = st.sidebar.checkbox(
            feature.replace('has', '').replace('is', '').replace('_', ' ').title(),
            value=False
        )
    
    return inputs

def make_prediction(model, scaler, feature_names, inputs):
    """Make prediction using the model"""
    # Create feature vector in correct order
    feature_vector = np.array([inputs.get(f, 0) for f in feature_names]).reshape(1, -1)
    
    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)
    
    # Predict
    prediction = model.predict(feature_vector_scaled)[0]
    
    return prediction

def main():
    """Main app function"""
    # Header
    st.markdown('<h1 class="main-header">🏠 Paris Housing Price Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, scaler, feature_names, metrics = load_model()
    
    # Sidebar with inputs
    st.sidebar.title("📊 Input Features")
    st.sidebar.markdown("Adjust the features below to predict the housing price.")
    
    inputs = get_feature_inputs(feature_names)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Prediction")
        
        # Prediction button
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            prediction = make_prediction(model, scaler, feature_names, inputs)
            
            # Display prediction
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Price</h2>
                    <div class="prediction-value">€{prediction:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional info
            st.info(f"💰 This prediction is based on a linear regression model with R² = {metrics['test_r2']:.4f}")
    
    with col2:
        st.header("📈 Model Performance")
        st.metric("Test R² Score", f"{metrics['test_r2']:.4f}")
        st.metric("Test RMSE", f"€{metrics['test_rmse']:,.2f}")
        st.metric("Cross-Validation R²", 
                 f"{metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        
        st.markdown("---")
        st.subheader("📊 Model Info")
        st.write(f"**Features Used:** {len(feature_names)}")
        st.write(f"**Model Type:** Multi-Linear Regression")
    
    # Feature importance visualization
    st.markdown("---")
    st.header("🔍 Feature Importance")
    
    try:
        feature_importance = pd.read_csv('models/feature_importance.csv')
        top_features = feature_importance.head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_features, y='Feature', x='Abs_Coefficient', ax=ax, palette='viridis')
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_title('Top 10 Most Important Features')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    except:
        st.info("Feature importance data not available. Run the training script to generate it.")
    
    # Sample predictions section
    st.markdown("---")
    st.header("💡 Quick Examples")
    
    col1, col2, col3 = st.columns(3)
    
    example_properties = [
        {
            "name": "Small Apartment",
            "squareMeters": 30000,
            "numberOfRooms": 20,
            "floors": 30,
            "hasPool": False,
            "hasYard": False
        },
        {
            "name": "Medium House",
            "squareMeters": 60000,
            "numberOfRooms": 50,
            "floors": 50,
            "hasPool": True,
            "hasYard": True
        },
        {
            "name": "Large Villa",
            "squareMeters": 90000,
            "numberOfRooms": 80,
            "floors": 70,
            "hasPool": True,
            "hasYard": True
        }
    ]
    
    for i, example in enumerate(example_properties):
        with [col1, col2, col3][i]:
            st.subheader(example["name"])
            example_inputs = inputs.copy()
            example_inputs.update(example)
            # Fill in default values for other features
            for feature in feature_names:
                if feature not in example_inputs:
                    example_inputs[feature] = inputs.get(feature, 0)
            
            example_pred = make_prediction(model, scaler, feature_names, example_inputs)
            st.metric("Estimated Price", f"€{example_pred:,.2f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Built with ❤️ using Streamlit | Paris Housing Price Prediction Model</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

