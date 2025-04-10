from ml_methods.random_forest import prepare_data, train_rf, evaluate

X_train, y_train, encoder, _ = prepare_data("data/train", use_pca=True)
X_test, y_test, _, _ = prepare_data("data/test", use_pca=True)

clf = train_rf(X_train, y_train)
evaluate(clf, X_test, y_test, encoder)
