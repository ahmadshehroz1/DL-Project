# train_student_kd.py
import torch, timm
from torch.utils.data import DataLoader
from scripts.dataset import LabeledDataset
from scripts.utils import get_transforms
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

class DistillLoss(torch.nn.Module):
    def __init__(self, T=4.0, alpha=0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        kd = F.kl_div(F.log_softmax(student_logits/self.T, dim=1),
                      F.softmax(teacher_logits/self.T, dim=1),
                      reduction="batchmean")*(self.T*self.T)
        ce = self.ce(student_logits, labels)
        return self.alpha*kd + (1-self.alpha)*ce

def train_student(train_csvs, val_csvs, device="cuda"):
    BATCH_SIZE = 16
    EPOCHS = 5
    IMG_DIR = "data/index"

    train_transform, val_transform = get_transforms(224)
    
    print("Loading Training Data for Student...")
    train_dataset = LabeledDataset(IMG_DIR, train_csvs, transform=train_transform)
    
    print("Loading Validation Data for Student...")
    val_dataset = LabeledDataset(IMG_DIR, val_csvs, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Load Teacher
    teacher = timm.create_model("resnet50", pretrained=False, num_classes=len(train_csvs))
    teacher.load_state_dict(torch.load("models/teacher_finetuned_resnet50.pth"))
    teacher.to(device).eval()

    # Create Student
    student = timm.create_model("resnet18", pretrained=False, num_classes=len(train_csvs))
    student.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    scaler = GradScaler()
    criterion = DistillLoss()

    print("=== Student Training (KD) ===")
    for epoch in range(EPOCHS):
        # Training
        student.train()
        total_loss, correct, total = 0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            
            optimizer.zero_grad()
            with autocast():
                student_logits = student(imgs)
                loss = criterion(student_logits, teacher_logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * imgs.size(0)
            preds = student_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=correct/total)
            
        train_acc = correct / total

        # Validation
        student.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                student_logits = student(imgs)
                teacher_logits = teacher(imgs)
                loss = criterion(student_logits, teacher_logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = student_logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        
        # Prevent division by zero if val set is empty
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"Epoch {epoch+1} Summary - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(student.state_dict(), "models/student_resnet18.pth")
    print("Student model saved!")