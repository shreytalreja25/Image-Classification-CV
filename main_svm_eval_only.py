import joblib
from ml_methods.sift_svm import prepare_data, evaluate

print("Loading saved SVM model for evaluation only...")

# Load data from test set
X_test, y_test, label_encoder, _ = prepare_data("data/test", use_pca=True)

# Load saved model
clf = joblib.load("models/svm_classifier.joblib")

# Run evaluation and log report
evaluate(clf, X_test, y_test, label_encoder)
