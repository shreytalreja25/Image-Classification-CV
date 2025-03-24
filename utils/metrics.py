from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_classification(y_true, y_pred, class_names=None, verbose=True):
    """
    Evaluate classification performance and optionally print a report.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if verbose:
        print("\n📊 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, figsize=(10, 8), cmap="Blues", title="Confusion Matrix", save_path=None):
    """
    Plot and optionally save a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📁 Saved confusion matrix to {save_path}")
    else:
        plt.show()
