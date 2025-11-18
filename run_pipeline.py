"""
Main pipeline script to run the complete ML workflow
Run this script to execute the entire pipeline from data exploration to model training
"""

import subprocess
import sys
import os

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 60)
    print(f"RUNNING: {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {description}: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_path}")
        return False

def main():
    """Run the complete pipeline"""
    print("\n" + "=" * 60)
    print("PARIS HOUSING PRICE PREDICTION - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists('data/ParisHousing.csv'):
        print("\n✗ Error: data/ParisHousing.csv not found!")
        print("Please ensure the dataset is in the data/ directory.")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Data exploration and preprocessing
    success = run_script(
        'src/explore_and_preprocess.py',
        'Data Exploration and Preprocessing'
    )
    
    if not success:
        print("\n✗ Pipeline stopped due to errors in preprocessing step.")
        return
    
    # Step 2: Model training
    success = run_script(
        'src/train_model.py',
        'Model Training and Evaluation'
    )
    
    if not success:
        print("\n✗ Pipeline stopped due to errors in training step.")
        return
    
    # Success message
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the generated visualizations in the models/ directory")
    print("2. Check model metrics in models/model_metrics.json")
    print("3. Run the Streamlit app: streamlit run app.py")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

