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

def run_xgb():
    print("\nRunning ML Pipeline (SIFT + XGBoost)...\n")
    os.system("python main_xgb.py")

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

def run_finetune_llm():
    print("\n🧠 Finetuning CLIP LLM on aerial landscape dataset...\n")
    os.system("python LLM_method\clip_finetune.py")

def run_llm_inference():
    print("\n🤖 Running CLIP LLM inference on 10 images per category...\n")
    os.system("python LLM_method\clip_infer_save.py")

def show_info():
    print("\nCOMP9517 CV Group Project")
    print("-" * 30)
    print("Task: Classify 15 landscape categories from aerial images")
    print("Models:")
    print("  • SIFT + SVM")
    print("  • SIFT + SGDClassifier")
    print("  • SIFT + Random Forest")
    print("  • SIFT + XGBoost")
    print("  • ResNet-18")
    print("  • EfficientNet-B0")
    print("  • MobileNetV2")
    print("  • CLIP LLM (Finetuned on aerial images)")
    print("\nVisualizations:")
    print("  • Grad-CAM for all DL models")
    print("\nReports:")
    print("  • All reports & confusion matrices saved in 'results/'")
    print("Due: 25 April 2025\n")

def main_menu():
    while True:
        print("======== MENU ========")
        print("1.  Split Dataset (Run Once)")
        print("2.  Run ML Pipeline (SIFT + SVM)")
        print("3.  Run ML Pipeline (SIFT + SGDClassifier)")
        print("4.  Run ML Pipeline (SIFT + Random Forest)")
        print("5.  Run ML Pipeline (SIFT + XGBoost)")
        print("6.  Run DL Pipeline (ResNet-18)")
        print("7.  Run DL Pipeline (EfficientNet-B0)")
        print("8.  Run DL Pipeline (MobileNetV2)")
        print("9.  Generate Grad-CAM Visualizations")
        print("10. Finetune CLIP LLM Model")
        print("11. Run CLIP LLM Inference (10 images/class)")
        print("12. Project Info")
        print("13. Exit")

        choice = input("Enter your choice (1–13): ")

        if choice == '1':
            split_data()
        elif choice == '2':
            run_ml()
        elif choice == '3':
            run_sgd()
        elif choice == '4':
            run_rf()
        elif choice == '5':
            run_xgb()
        elif choice == '6':
            run_dl()
        elif choice == '7':
            run_efficientnet()
        elif choice == '8':
            run_mobilenet()
        elif choice == '9':
            run_gradcam()
        elif choice == '10':
            run_finetune_llm()
        elif choice == '11':
            run_llm_inference()
        elif choice == '12':
            show_info()
        elif choice == '13':
            print("Exiting...")
            time.sleep(1)
            break
        else:
            print("Invalid choice. Try again.\n")

if __name__ == "__main__":
    main_menu()
