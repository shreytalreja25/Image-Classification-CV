# filename: random_forest.py
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from datetime import datetime
from tqdm import tqdm
from utils.metrics import evaluate_classification, plot_confusion_matrix
import joblib

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_sift_features(image_path, max_descriptors=200):
    """
    Reads an image in grayscale, extracts SIFT descriptors,
    and returns a flattened array of shape (max_descriptors * 128).
    If descriptors < max_descriptors, zero-pad. If more, truncate.
    """
    try:
        img = cv2.imread(image_path, 0)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        # If image is blank or featureless, return an all-zero vector
        if descriptors is None:
            return np.zeros((max_descriptors, 128)).flatten()

        # Truncate or pad to max_descriptors
        if descriptors.shape[0] > max_descriptors:
            descriptors = descriptors[:max_descriptors]
        elif descriptors.shape[0] < max_descriptors:
            pad = np.zeros((max_descriptors - descriptors.shape[0], 128))
            descriptors = np.vstack((descriptors, pad))

        return descriptors.flatten()

    except:
        # If something fails (e.g. corrupt image), return None
        return None

def prepare_data(
    data_dir, 
    max_descriptors=200, 
    use_pca=True, 
    pca_dim=256,
    force_recalc=False
):
    """
    Extracts SIFT features for each image in data_dir, optionally applies PCA,
    and returns X, y, label_encoder, and pca_obj (if use_pca=True).
    
    Parameters:
    - force_recalc: if True, ignores & deletes any existing cache to do a fresh run.
    """
    # Example cache filename
    cache_name = f"{'pca' if use_pca else 'nopca'}_{os.path.basename(data_dir)}_rf_sift_{max_descriptors}_dim{pca_dim}.npz"
    cache_path = os.path.join(CACHE_DIR, cache_name)

    # If we suspect incomplete or stale data, we can forcibly remove the old cache
    if force_recalc and os.path.exists(cache_path):
        os.remove(cache_path)

    if os.path.exists(cache_path):
        print(f"\n[RF] Loading cached features from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        X = cache["X"]
        y = cache["y"]
        labels = cache["labels"]

        label_encoder = LabelEncoder()
        label_encoder.classes_ = labels

        pca_obj = cache["pca"] if "pca" in cache.files else None
        return X, y, label_encoder, pca_obj

    print(f"\n[RF] Extracting features from: {data_dir}")
    X, y = [], []
    label_encoder = LabelEncoder()

    category_folders = [d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))]
    
    # We'll do a quick expected file count to help diagnose missing data
    total_files = 0
    for c in category_folders:
        label_path = os.path.join(data_dir, c)
        total_files += len(os.listdir(label_path))

    print(f"Expected to find ~{total_files} images in {data_dir}...")

    for label in category_folders:
        label_path = os.path.join(data_dir, label)
        files = os.listdir(label_path)
        print(f" -> {label} with {len(files)} images")

        for file in tqdm(files, desc=f"   Extracting [{label}]", ncols=80):
            image_path = os.path.join(label_path, file)
            feat = extract_sift_features(image_path, max_descriptors)
            if feat is not None:
                X.append(feat)
                y.append(label)
            # else we skip that image entirely

    X = np.array(X)
    y = label_encoder.fit_transform(y)
    labels = label_encoder.classes_

    # Show how many images were successfully processed
    print(f"[RF] Processed {X.shape[0]} images out of {total_files} available.")
    if X.shape[0] < total_files:
        print("   [!] Warning: Some images may be corrupted or had no features.\n")

    pca_obj = None
    if use_pca:
        print(f"[RF] Applying PCA (dim={pca_dim}) to reduce dimensionality...")
        pca_obj = PCA(n_components=pca_dim)
        X = pca_obj.fit_transform(X)

    print(f"[RF] Caching features to {cache_path}")
    np.savez_compressed(cache_path, X=X, y=y, labels=labels, pca=pca_obj)

    return X, y, label_encoder, pca_obj

def train_rf(X_train, y_train):
    """
    Train a Random Forest classifier with some tuned hyperparameters
    and save the model to models/rf_classifier.joblib
    """
    print("\n🌲 Training Random Forest classifier with updated hyperparams...")
    clf = RandomForestClassifier(
        # Increased to 300 trees for better ensemble performance
        n_estimators=300,            
        # Medium depth to limit overfitting & speed up training
        max_depth=30,                
        # Larger min_samples_split to help reduce overfitting
        min_samples_split=4,         
        # 'sqrt' is typical for image data; can also try 'log2'
        max_features='sqrt',        
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("✅ Training complete.")

    os.makedirs("models", exist_ok=True)
    model_path = "models/rf_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"💾 Model saved to: {model_path}")
    return clf

def evaluate(clf, X_test, y_test, label_encoder):
    """
    Evaluate the trained classifier on test data, 
    save classification report and confusion matrix to disk.
    """
    print("\n🔍 Evaluating Random Forest model...")
    preds = clf.predict(X_test)
    class_names = label_encoder.classes_
    
    # Summarize metrics
    from sklearn.metrics import classification_report
    report = classification_report(y_test, preds, target_names=class_names, digits=4)

    # Evaluate_classification is your custom metric aggregator
    metrics = evaluate_classification(y_test, preds, class_names, verbose=False)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    result_dir = os.path.join("results", "ML_results", f"rf_eval_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Save textual report
    report_file = os.path.join(result_dir, "report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Timestamp: {timestamp}\n")
        f.write("Model: SIFT + Random Forest (Tuned)\n\n")
        f.write("Summary Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nFull Classification Report:\n")
        f.write(report)

    print(f"📄 Report saved to: {report_file}")

    # Confusion matrix plot
    cm_path = os.path.join(result_dir, "confmat.png")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=preds,
        class_names=class_names,
        normalize=True,
        title="SIFT + Random Forest (Tuned) Confusion Matrix",
        save_path=cm_path
    )
    print(f"🖼️ Confusion matrix saved to: {cm_path}")

    # Also show the classification report in console
    print("\n📊 Classification Report:")
    print(report)
