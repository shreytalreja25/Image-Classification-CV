# main_densenet.py

import os
from dl_methods.densenet import run_densenet

def main():
    data_dir     = "data"
    train_dir    = os.path.join(data_dir, "train")
    test_dir     = os.path.join(data_dir, "test")
    num_classes  = 15
    batch_size   = 32
    num_epochs   = 10
    learning_rate= 1e-4
    save_model   = "models/densenet121.pth"
    results_root = "results/DL_results"

    run_densenet(
        train_dir=train_dir,
        test_dir=test_dir,
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_model_path=save_model,
        results_root=results_root
    )

if __name__ == "__main__":
    main()
