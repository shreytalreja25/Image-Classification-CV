# ml_methods/xgboost_classifier.py

import os
import shutil
import joblib
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from utils.metrics import evaluate_classification, plot_confusion_matrix
from ml_methods.random_forest import prepare_data as base_prepare_data, CACHE_DIR

def _clear_cache(data_dir, max_descriptors):
    """
    Remove any existing cache file for this dataset so features are re-extracted.
    """
    # Cache filenames follow: "{pca|nopca}_{basename}_rf_sift_{max_descriptors}.npz"
    basename = os.path.basename(data_dir)
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith(f"{basename}_rf_sift_{max_descriptors}.npz"):
            os.remove(os.path.join(CACHE_DIR, fname))

def prepare_data(data_dir, max_descriptors=200, use_pca=True, pca_dim=256):
    """
    Force fresh SIFT+PCA extraction by clearing cache first.
    """
    _clear_cache(data_dir, max_descriptors)
    return base_prepare_data(
        data_dir,
        max_descriptors=max_descriptors,
        use_pca=use_pca,
        pca_dim=pca_dim
    )

def train_xgb(X_train, y_train):
    """
    Train XGBoost with verbose progress updates.
    """
    print("\n🚀 Training XGBoost classifier with progress updates...\n")
    total_trees = 300

    clf = XGBClassifier(
        n_estimators=total_trees,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        use_label_encoder=False,
        verbosity=0  # silence internal logs; we'll use fit(verbose)
    )

    # prints a line every 10 trees
    clf.fit(X_train, y_train, verbose=10)
    print("✅ Training complete.")

    os.makedirs("models", exist_ok=True)
    model_path = "models/xgb_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"💾 Model saved to: {model_path}")
    return clf

def evaluate(clf, X_test, y_test, label_encoder):
    """
    Evaluate and save report, confusion matrix, and append summary.
    """
    print("\n📊 Evaluating XGBoost model...")
    preds = clf.predict(X_test)
    class_names = label_encoder.classes_
    report = classification_report(y_test, preds, target_names=class_names, digits=4)
    metrics = evaluate_classification(y_test, preds, class_names, verbose=False)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    result_dir = os.path.join("results", "ML_results", f"xgb_eval_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Detailed report
    report_path = os.path.join(result_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Timestamp: {timestamp}\n")
        f.write("Model: SIFT + XGBoost\n\n")
        f.write("Summary Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nFull Classification Report:\n")
        f.write(report)
    print(f"📄 Report saved to: {report_path}")

    # Confusion matrix
    conf_path = os.path.join(result_dir, "confmat.png")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=preds,
        class_names=class_names,
        normalize=True,
        title="SIFT + XGBoost Confusion Matrix",
        save_path=conf_path
    )
    print(f"🖼️ Confusion matrix saved to: {conf_path}")

    # Append summary
    summary_path = os.path.join("results", "summary_report.txt")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"\n----- XGBoost Evaluation ({timestamp}) -----\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"📌 Summary appended to: {summary_path}")

    # Console output
    print("\n📊 Classification Report:")
    print(report)
