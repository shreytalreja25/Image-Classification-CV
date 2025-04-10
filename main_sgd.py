from ml_methods.sgd_svm import prepare_data, train_sgd, evaluate

if __name__ == "__main__":
    print("\n🔍 Starting SIFT + SGDClassifier Pipeline...\n")

    # Load training data
    X_train, y_train, encoder, _ = prepare_data("data/train", use_pca=True)

    # Load test data — ensure label_encoder is reused for evaluation
    X_test, y_test, _, _ = prepare_data("data/test", use_pca=True)

    # Train the model
    clf = train_sgd(X_train, y_train)

    # Evaluate and generate report
    evaluate(clf, X_test, y_test, encoder)
