import os
import sys
import time

def run_ml():
    print("\nRunning Machine Learning Pipeline (SIFT + SVM)...\n")
    os.system("python main_ml.py")

def run_sgd():
    print("\nRunning ML Pipeline (SIFT + SGDClassifier)...\n")
    os.system("python main_sgd.py")

def run_rf():
    print("\nRunning ML Pipeline (SIFT + Random Forest)...\n")
    os.system("python main_rf.py")

def run_dl():
    print("\nRunning Deep Learning Pipeline (ResNet-18)...\n")
    os.system("python main_dl.py")

def run_efficientnet():
    print("\nRunning EfficientNet-B0 Pipeline...\n")
    os.system("python main_efficientnet.py")

def run_mobilenet():
    print("\nRunning MobileNetV2 Pipeline...\n")
    os.system("python main_mobilenet.py")

def split_data():
    print("\nSplitting Dataset into Train/Test...\n")
    from utils.dataset_utils import split_dataset
    split_dataset('archive/Aerial_Landscapes', 'data')
    print("\nDataset split completed.\n")

def run_gradcam():
    print("\nRunning Grad-CAM Visualizations...\n")
    print("1. ResNet-18")
    print("2. EfficientNet-B0")
    print("3. MobileNetV2")
    model_choice = input("Choose model for Grad-CAM (1–3): ")

    from dl_methods.generate_all_gradcams import (
        run_batch_gradcam,
        run_batch_gradcam_efficientnet,
        run_batch_gradcam_mobilenet
    )

    if model_choice == '1':
        run_batch_gradcam()
    elif model_choice == '2':
        run_batch_gradcam_efficientnet()
    elif model_choice == '3':
        run_batch_gradcam_mobilenet()
    else:
        print("Invalid choice. Returning to main menu.\n")

def show_info():
    print("\nCOMP9517 CV Group Project")
    print("-" * 30)
    print("Task: Classify 15 landscape categories from aerial images")
    print("Models:")
    print("  • SIFT + SVM")
    print("  • SIFT + SGDClassifier")
    print("  • SIFT + Random Forest")
    print("  • ResNet-18")
    print("  • EfficientNet-B0")
    print("  • MobileNetV2")
    print("Visualizations: Grad-CAM for all DL models")
    print("Due: 25 April 2025\n")

def main_menu():
    while True:
        print("======== MENU ========")
        print("1.  Split Dataset (Run Once)")
        print("2.  Run ML Pipeline (SIFT + SVM)")
        print("3.  Run ML Pipeline (SIFT + SGDClassifier)")
        print("4.  Run ML Pipeline (SIFT + Random Forest)")
        print("5.  Run DL Pipeline (ResNet-18)")
        print("6.  Run DL Pipeline (EfficientNet-B0)")
        print("7.  Run DL Pipeline (MobileNetV2)")
        print("8.  Generate Grad-CAM Visualizations")
        print("9.  Project Info")
        print("10. Exit")

        choice = input("Enter your choice (1–10): ")

        if choice == '1':
            split_data()
        elif choice == '2':
            run_ml()
        elif choice == '3':
            run_sgd()
        elif choice == '4':
            run_rf()
        elif choice == '5':
            run_dl()
        elif choice == '6':
            run_efficientnet()
        elif choice == '7':
            run_mobilenet()
        elif choice == '8':
            run_gradcam()
        elif choice == '9':
            show_info()
        elif choice == '10':
            print("Exiting...")
            time.sleep(1)
            break
        else:
            print("Invalid choice. Try again.\n")

if __name__ == "__main__":
    main_menu()
