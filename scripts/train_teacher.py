# train_teacher.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from scripts.dataset import LabeledDataset, UnlabeledDataset
from scripts.utils import get_transforms, DataAugmentationDINO
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os

# ---------------- DINO Head ----------------
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # Handle list of crops or single tensor
        if isinstance(x, list):
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)
            start_idx = 0
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
            return self.head(output)
        else:
            return self.head(self.backbone(x))

# ---------------- DINO Loss ----------------
class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        student_output: (n_crops * B, out_dim)
        teacher_output: (2 * B, out_dim) -> Only global crops
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(len(student_output) // teacher_output.shape[0] * 2) # split per view

        # Teacher centering and sharpening
        temp = self.teacher_temp
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:

                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# ---------------- Utils for DINO ----------------
class Utils:
    @staticmethod
    def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        schedule = np.concatenate((warmup_schedule, schedule))
        return schedule

# ---------------- SSL Teacher ----------------

# Update the arguments to accept csv_whitelist
def train_teacher_ssl(device="cuda", csv_whitelist=None):
    BATCH_SIZE = 32
    EPOCHS = 5 
    IMG_DIR = "data/index"
    OUT_DIM = 4096 

    dino_transform = DataAugmentationDINO(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=2
    )

    # Pass the whitelist to the dataset
    dataset = UnlabeledDataset(IMG_DIR, transform=dino_transform, csv_files=csv_whitelist)
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)


    # 2. Architectures (Student & Teacher)
    def create_model():
        backbone = timm.create_model("resnet50", pretrained=False, num_classes=0) 
        head = DINOHead(backbone.num_features, OUT_DIM)
        return MultiCropWrapper(backbone, head)

    student = create_model().to(device)
    teacher = create_model().to(device)
    
    # Initialize teacher with student weights
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False 

    # 3. Optimization
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.0005, weight_decay=0.04)
    criterion = DINOLoss(OUT_DIM).to(device)
    scaler = GradScaler()
    
    # Momentum scheduler for teacher update
    momentum_schedule = Utils.cosine_scheduler(0.996, 1, EPOCHS, len(loader))

    print("=== SSL Teacher Training (DINO Real) ===")
    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for it, images in enumerate(loop):
            # images is a list of tensors
            images = [im.to(device, non_blocking=True) for im in images]
            
            # --- Forward ---
            with autocast():
                # Student processes ALL crops (global + local)
                student_output = student(images)
                # Teacher processes ONLY GLOBAL crops (first 2)
                with torch.no_grad():
                    teacher_output = teacher(images[:2])
                
                loss = criterion(student_output, teacher_output)

            # --- Backward ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Update Teacher (EMA) ---
            with torch.no_grad():
                m = momentum_schedule[it + epoch * len(loader)] 
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss/(it+1))

    torch.save(teacher.backbone.state_dict(), "models/teacher_ssl_resnet50.pth")
    print("SSL Teacher backbone saved!")

# ---------------- Fine-tune Teacher ----------------

def train_teacher_labeled(train_csvs, val_csvs, device="cuda"):
    BATCH_SIZE = 16
    EPOCHS = 5  
    IMG_DIR = "data/index"

    train_transform, val_transform = get_transforms(224)
    
    print("Loading Training Data...")
    train_dataset = LabeledDataset(IMG_DIR, train_csvs, transform=train_transform)
    
    print("Loading Validation Data...")
    val_dataset = LabeledDataset(IMG_DIR, val_csvs, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Load SSL Pretrained Backbone
    model = timm.create_model("resnet50", pretrained=False, num_classes=len(train_csvs))
    
    # Check if SSL weights exist and load
    if os.path.exists("models/teacher_ssl_resnet50.pth"):
        print("Loading SSL weights...")
        state_dict = torch.load("models/teacher_ssl_resnet50.pth")
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded: {msg}")
    
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    print("=== Fine-tuning Teacher ===")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            loop.set_postfix(acc=correct/total)
        
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        
        # Avoid division by zero if val set is empty
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "models/teacher_finetuned_resnet50.pth")
    print("Fine-tuned Teacher saved!")