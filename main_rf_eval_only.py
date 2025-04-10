from ml_methods.random_forest import prepare_data, evaluate
import joblib
import os

# Load cached test data
X_test, y_test, encoder, _ = prepare_data("data/test", use_pca=True)

# Load trained model
model_path = "models/rf_classifier.joblib"
if not os.path.exists(model_path):
    print(f"❌ Trained model not found at {model_path}. Please run main_rf.py first.")
    exit()

print("\n🧪 Loading saved Random Forest model for evaluation only...")
clf = joblib.load(model_path)

# Evaluate
evaluate(clf, X_test, y_test, encoder)
