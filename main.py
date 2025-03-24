import os
import sys
import time

def run_ml():
    print("\nRunning Machine Learning Pipeline (SIFT + SVM)...\n")
    os.system("python main_ml.py")

def run_sgd():
    print("\nRunning ML Pipeline (SIFT + SGDClassifier)...\n")
    os.system("python main_sgd.py")

def run_dl():
    print("\nRunning Deep Learning Pipeline (ResNet-18)...\n")
    os.system("python main_dl.py")

def run_efficientnet():
    print("\nRunning EfficientNet-B0 Pipeline...\n")
    os.system("python main_efficientnet.py")

def split_data():
    print("\nSplitting Dataset into Train/Test...\n")
    from utils.dataset_utils import split_dataset
    split_dataset('archive/Aerial_Landscapes', 'data')
    print("\nDataset split completed.\n")

def run_gradcam():
    print("\nRunning Grad-CAM on test images...\n")
    from dl_methods.generate_all_gradcams import run_batch_gradcam
    run_batch_gradcam()

def show_info():
    print("\nCOMP9517 CV Group Project")
    print("-" * 30)
    print("Task: Classify 15 landscape categories from aerial images")
    print("Using: SIFT+SVM, SIFT+SGDClassifier, ResNet-18, EfficientNet")
    print("With: Grad-CAM Visualizations")
    print("Report & video due: 25 April 2025\n")

def main_menu():
    while True:
        print("======== MENU ========")
        print("1. Split Dataset (Run Once)")
        print("2. Run ML Pipeline (SIFT + SVM)")
        print("3. Run DL Pipeline (ResNet-18)")
        print("4. Run EfficientNet-B0")
        print("5. Generate Grad-CAM Visualizations")
        print("6. Project Info")
        print("7. Exit")
        print("8. Run ML Pipeline (SIFT + SGDClassifier)")

        choice = input("Enter your choice (1–8): ")

        if choice == '1':
            split_data()
        elif choice == '2':
            run_ml()
        elif choice == '3':
            run_dl()
        elif choice == '4':
            run_efficientnet()
        elif choice == '5':
            run_gradcam()
        elif choice == '6':
            show_info()
        elif choice == '7':
            print("Exiting...")
            time.sleep(1)
            break
        elif choice == '8':
            run_sgd()
        else:
            print("Invalid choice. Try again.\n")

if __name__ == "__main__":
    main_menu()
