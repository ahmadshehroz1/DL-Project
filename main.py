# main.py
import torch
import multiprocessing
multiprocessing.freeze_support()

from scripts.train_teacher import train_teacher_ssl, train_teacher_labeled
from scripts.train_student_kd import train_student


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    train_files = {
        "negative": "data/negative_train.csv",
        "positive": "data/positive_train.csv",
        "surprise": "data/surprise_train.csv"
    }

    test_files = {
        "negative": "data/negative_test.csv",
        "positive": "data/positive_test.csv",
        "surprise": "data/surprise_test.csv"
    }

    # 2. SSL Teacher Training
    print("--- Starting Inductive SSL Training (Train Set Only) ---")
    train_teacher_ssl(device=device, csv_whitelist=train_files) 

    # 3. Fine-Tuning
    print("--- Fine Tuning Teacher ---")
    train_teacher_labeled(train_files, test_files, device=device)

    # 4. Distillation
    print("--- Distilling to Student ---")
    train_student(train_files, test_files, device=device)

if __name__ == "__main__":
    main()

