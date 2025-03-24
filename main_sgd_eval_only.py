import joblib
from ml_methods.sgd_svm import prepare_data, evaluate

print("Loading SGDClassifier for evaluation only...")

# Load data
_, X_test, y_test, label_encoder, _ = prepare_data("data/test", use_pca=True)

# Load model
clf = joblib.load("models/sgd_classifier.joblib")

# Evaluate
evaluate(clf, X_test, y_test, label_encoder)
