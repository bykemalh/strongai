import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Tekrarlanabilirlik için seed ayarı
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# GPU kullanımı kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Veri yolları
DATA_ROOT = "dataset"
NORMAL_DIR = os.path.join(DATA_ROOT, "normal")
ISKEMI_DIR = os.path.join(DATA_ROOT, "iskemi")
KANAMA_DIR = os.path.join(DATA_ROOT, "kanama")
ANNOTATIONS_FILE = os.path.join(DATA_ROOT, "overlay", "annotations.json")

# Görüntü dönüşümleri
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Özel veri seti sınıfı
class StrokeDataset(Dataset):
    def __init__(self, normal_dir, iskemi_dir, kanama_dir, transform=None):
        self.transform = transform
        
        # Normal görüntülerin yolları
        self.normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.png')]
        
        # İskemi görüntülerinin yolları
        self.iskemi_images = [os.path.join(iskemi_dir, f) for f in os.listdir(iskemi_dir) if f.endswith('.png')]
        
        # Kanama görüntülerinin yolları
        self.kanama_images = [os.path.join(kanama_dir, f) for f in os.listdir(kanama_dir) if f.endswith('.png')]
        
        # Tüm görüntü yolları ve etiketleri birleştirme
        self.all_images = self.normal_images + self.iskemi_images + self.kanama_images
        self.labels = [0] * len(self.normal_images) + [1] * (len(self.iskemi_images) + len(self.kanama_images))
        
        # Görüntülerin karıştırılması
        combined = list(zip(self.all_images, self.labels))
        random.shuffle(combined)
        self.all_images, self.labels = zip(*combined)
        
        print(f"Normal görüntü sayısı: {len(self.normal_images)}")
        print(f"İskemi görüntü sayısı: {len(self.iskemi_images)}")
        print(f"Kanama görüntü sayısı: {len(self.kanama_images)}")
        print(f"Toplam görüntü sayısı: {len(self.all_images)}")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Veri kümesini train ve test olarak bölme
def split_dataset(dataset, test_ratio=0.1):
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

# Özel CNN modeli tasarımı
class StrokeDetectionModel(nn.Module):
    def __init__(self):
        super(StrokeDetectionModel, self).__init__()
        
        # Konvolüsyon katmanları
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling katmanı
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # İki sınıf: inme var (1) veya yok (0)
        
    def forward(self, x):
        # Konvolüsyon blokları
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 256x256
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 128x128
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 64x64
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 32x32
        
        # Flatten
        x = x.view(-1, 256 * 32 * 32)
        x = self.dropout1(x)
        
        # Tam bağlantılı katmanlar
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Model eğitim fonksiyonu
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Eğitim"):
        images, labels = images.to(device), labels.to(device)
        
        # Gradyanları sıfırla
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass ve optimize et
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Doğruluk hesaplama
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Model test fonksiyonu
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# F1 skoru hesaplama
def calculate_f1_score(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='weighted')

# Veri yükleme ve ön işleme
full_dataset = StrokeDataset(NORMAL_DIR, ISKEMI_DIR, KANAMA_DIR, transform=transform)
train_dataset, test_dataset = split_dataset(full_dataset)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# Model oluşturma
model = StrokeDetectionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# Eğitim değişkenlerini izleme
epochs = 50
train_losses = []
test_losses = []
train_accs = []
test_accs = []
f1_scores = []
best_f1 = 0.0

# Eğitim döngüsü
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Eğitim
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Test
    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # F1 skoru hesaplama
    f1 = calculate_f1_score(all_labels, all_preds)
    f1_scores.append(f1)
    
    # Öğrenme oranını ayarla
    scheduler.step(test_loss)
    
    # Sonuçları yazdır
    print(f"Eğitim Kaybı: {train_loss:.4f}, Eğitim Doğruluğu: {train_acc:.2f}%")
    print(f"Test Kaybı: {test_loss:.4f}, Test Doğruluğu: {test_acc:.2f}%")
    print(f"F1 Skoru: {f1:.4f}")
    
    # En iyi modeli kaydet
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'test7_model.pth')
        print(f"En iyi model kaydedildi! F1 skoru: {best_f1:.4f}")
    
    print("-" * 50)

# Eğitim sonuçlarını görselleştirme
plt.figure(figsize=(15, 5))

# Kayıp grafiği
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Eğitim')
plt.plot(test_losses, label='Test')
plt.title('Epoch başına Kayıp')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

# Doğruluk grafiği
plt.subplot(1, 3, 2)
plt.plot(train_accs, label='Eğitim')
plt.plot(test_accs, label='Test')
plt.title('Epoch başına Doğruluk')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk (%)')
plt.legend()

# F1 skoru grafiği
plt.subplot(1, 3, 3)
plt.plot(f1_scores)
plt.title('Epoch başına F1 Skoru')
plt.xlabel('Epoch')
plt.ylabel('F1 Skoru')

plt.tight_layout()
plt.savefig('training_results_test7.png')
plt.show()

# En iyi modeli yükle ve test et
model.load_state_dict(torch.load('test7_model.pth'))
_, _, final_preds, final_labels = evaluate(model, test_loader, criterion, device)
final_f1 = calculate_f1_score(final_labels, final_preds)
print(f"En iyi modelin test üzerindeki F1 skoru: {final_f1:.4f}")

# Karışıklık matrisi
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'İnme Var'], yticklabels=['Normal', 'İnme Var'])
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Karışıklık Matrisi')
plt.savefig('confusion_matrix.png')
plt.show()

# ROC eğrisi ve AUC hesaplama
from sklearn.metrics import roc_curve, auc

model.eval()
probs = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs_batch = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        probs.extend(probs_batch)

fpr, tpr, _ = roc_curve(final_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('Alıcı İşletim Karakteristiği (ROC) Eğrisi')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

# Sonuçları dosyaya yazma
results = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'train_accs': train_accs,
    'test_accs': test_accs,
    'f1_scores': f1_scores,
    'best_f1': best_f1,
    'final_f1': final_f1,
    'roc_auc': roc_auc
}

import pickle
with open('training_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Eğitim tamamlandı ve sonuçlar kaydedildi!")