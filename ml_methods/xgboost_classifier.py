import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from utils.metrics import evaluate_classification, plot_confusion_matrix
import joblib
from xgboost import XGBClassifier
from ml_methods.random_forest import prepare_data as base_prepare_data
from tqdm import tqdm

def prepare_data(data_dir, use_pca=True):
    # Skip cache by forcing base_prepare_data to recalculate every time
    return base_prepare_data(
        data_dir,
        max_descriptors=200,
        use_pca=use_pca,
        pca_dim=256,
        force_recalc=True  # 🔥 forces fresh SIFT extraction
    )

def train_xgb(X_train, y_train):
    print("\n🚀 Training XGBoost classifier with tqdm progress bar...\n")
    
    # Manual tqdm bar since xgboost doesn't integrate natively
    total_trees = 300
    progress = tqdm(total=total_trees, desc="🌲 Boosting Trees", ncols=80)

    class TqdmCallback:
        def __init__(self, tqdm_bar):
            self.tqdm_bar = tqdm_bar
        def __call__(self, env):
            self.tqdm_bar.update(1)

    clf = XGBClassifier(
        n_estimators=total_trees,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        verbosity=0  # silence default logging
    )

    # Use tqdm callback
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        callbacks=[TqdmCallback(progress)]
    )
    progress.close()
    print("✅ Training complete.")

    os.makedirs("models", exist_ok=True)
    model_path = "models/xgb_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"💾 Model saved to: {model_path}")
    return clf

def evaluate(clf, X_test, y_test, label_encoder):
    print("\n📊 Evaluating XGBoost model...")
    preds = clf.predict(X_test)
    class_names = label_encoder.classes_
    report = classification_report(y_test, preds, target_names=class_names, digits=4)
    metrics = evaluate_classification(y_test, preds, class_names, verbose=False)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    result_dir = os.path.join("results", "ML_results", f"xgb_eval_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Save detailed report
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

    # Save confusion matrix
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

    print("\n📊 Classification Report:")
    print(report)
