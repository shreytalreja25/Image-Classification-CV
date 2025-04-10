import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from datetime import datetime
from tqdm import tqdm
from utils.metrics import evaluate_classification, plot_confusion_matrix
import time
import joblib
from tqdm import tqdm

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_sift_features(image_path, max_descriptors=500):
    try:
        img = cv2.imread(image_path, 0)
        sift = cv2.SIFT_create(nfeatures=max_descriptors)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            return np.zeros((max_descriptors, 128)).flatten()

        if descriptors.shape[0] > max_descriptors:
            descriptors = descriptors[:max_descriptors]
        else:
            pad = np.zeros((max_descriptors - descriptors.shape[0], 128))
            descriptors = np.vstack((descriptors, pad))

        return descriptors.flatten()
    except:
        return None

def prepare_data(data_dir, max_descriptors=500, use_pca=True, pca_dim=300):
    cache_name = f"{'pca' if use_pca else 'nopca'}_{os.path.basename(data_dir)}_sift_{max_descriptors}.npz"
    cache_path = os.path.join(CACHE_DIR, cache_name)

    if os.path.exists(cache_path):
        print(f"\nLoading cached features from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        return cache["X"], cache["y"], cache["labels"].tolist(), cache["pca"] if "pca" in cache.files else None

    print(f"\nExtracting features from: {data_dir}")
    X, y = [], []
    label_encoder = LabelEncoder()

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path): continue

        files = os.listdir(label_path)
        print(f"{label} - {len(files)} images")

        for file in tqdm(files, desc=f"   Extracting [{label}]", ncols=80):
            image_path = os.path.join(label_path, file)
            feat = extract_sift_features(image_path, max_descriptors)
            if feat is not None:
                X.append(feat)
                y.append(label)

    X = np.array(X)
    y = label_encoder.fit_transform(y)
    labels = label_encoder.classes_.tolist()

    pca_obj = None
    if use_pca:
        print("Applying PCA to reduce dimensionality...")
        pca_obj = PCA(n_components=pca_dim)
        # pca_obj = PCA()
        X = pca_obj.fit_transform(X)

    print(f"Caching features to {cache_path}")
    np.savez_compressed(cache_path, X=X, y=y, labels=labels, pca=pca_obj)

    return X, y, labels, pca_obj

def train_svm(X_train, y_train):
    print("\n⚙️ Training SVM classifier (SIFT + SVM)... This may take a few minutes.")

    start_time = time.time()

    # Verbose removed, tqdm used instead for cleaner CLI
    clf = SVC(kernel='linear', probability=True, verbose=False)

    # Fake progress bar for visual feel (since SVM doesn't expose training loop)
    with tqdm(total=1, desc="🧠 Fitting SVM", ncols=100) as pbar:
        clf.fit(X_train, y_train)
        pbar.update(1)

    duration = time.time() - start_time
    print(f"✅ Training complete. Time taken: {duration:.2f} seconds.")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/svm_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"💾 Model saved to: {model_path}")

    return clf

# def train_svm(X_train, y_train):
#     print("\nTraining SVM classifier (SIFT + SVM)... This may take a few minutes.")
#     start_time = time.time()
    
#     clf = SVC(kernel='linear', probability=True, verbose=1)
#     clf.fit(X_train, y_train)

#     duration = time.time() - start_time
#     print(f"Training complete. Time taken: {duration:.2f} seconds.")

#     # Save model
#     os.makedirs("models", exist_ok=True)
#     model_path = "models/svm_classifier.joblib"
#     joblib.dump(clf, model_path)
#     print(f"Model saved to: {model_path}")

#     return clf

def evaluate(clf, X_test, y_test, label_encoder):
    print("\nEvaluating model...")

    # label_encoder = LabelEncoder()
    # labels = label_encoder.classes_.tolist()


    preds = clf.predict(X_test)
    class_names = label_encoder
    report = classification_report(y_test, preds, target_names=class_names, digits=4)
    metrics = evaluate_classification(y_test, preds, class_names, verbose=False)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    result_dir = os.path.join("results", "ML_results", f"svm_eval_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    report_file = os.path.join(result_dir, "report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Timestamp: {timestamp}\n")
        f.write("Model: SIFT + SVM (Linear Kernel)\n")
        f.write("\nSummary Metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nFull Classification Report:\n")
        f.write(report)

    print(f"Report saved to: {report_file}")

    cm_path = os.path.join(result_dir, "confmat.png")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=preds,
        class_names=class_names,
        normalize=True,
        title="SIFT + SVM Confusion Matrix",
        save_path=cm_path
    )
    print(f"Confusion matrix saved to: {cm_path}")

    print(report)
