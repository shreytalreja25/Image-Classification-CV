# main_xgb.py
from ml_methods.xgboost_classifier import prepare_data, train_xgb, evaluate

X_train, y_train, encoder, _ = prepare_data("data/train", use_pca=True)
X_test, y_test, _, _ = prepare_data("data/test", use_pca=True)

clf = train_xgb(X_train, y_train)
evaluate(clf, X_test, y_test, encoder)
