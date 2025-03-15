import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
from scipy.ndimage import center_of_mass
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, classification_report, precision_recall_curve, average_precision_score
import seaborn as sns

# MONAI kütüphaneleri
from monai.networks.nets import DenseNet121
from monai.transforms import (
    LoadImage, Compose, Resize, ScaleIntensity, RandRotate, RandFlip,
    RandZoom, RandAffine, ToTensor, RandGaussianNoise, RandGaussianSmooth,
    RandAdjustContrast, RandHistogramShift, CenterSpatialCrop
)
from monai.data import pad_list_data_collate
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import compute_roc_auc
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image

# Tekrarlanabilirlik için seed ayarı
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=seed)

set_seed()

# GPU kullanımı kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Veri yolları
DATA_ROOT = os.path.expanduser("/home/kemal/Projects/TeknoFest/StrongAI/dataset")
NORMAL_DIR = os.path.join(DATA_ROOT, "0")  # İnme yok
STROKE_DIR = os.path.join(DATA_ROOT, "1")  # İnme var
TEST_ROOT = os.path.join(DATA_ROOT, "test")
TEST_NORMAL_DIR = os.path.join(TEST_ROOT, "0")  # Test - İnme yok
TEST_STROKE_DIR = os.path.join(TEST_ROOT, "1")  # Test - İnme var
ANNOTATIONS_JSON = os.path.join(DATA_ROOT, "annotations.json")  # Etiket JSON dosyası

# Sonuçların kaydedileceği klasör
RESULTS_DIR = os.path.join("/home/kemal/Projects/TeknoFest/StrongAI/Test16", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model kaydetme yolu
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Zaman damgalı model ismi oluştur
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = os.path.join(MODEL_DIR, f"stroke_detection_model_{timestamp}.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, f"best_stroke_detection_model_{timestamp}.pth")

# JSON etiket dosyasını yükle
def load_annotations(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Uyarı: {json_path} dosyası bulunamadı. Boş etiket sözlüğü kullanılıyor.")
        return {"iskeme": {}, "kanama": {}}

# Maske oluşturma fonksiyonu
def create_mask_from_points(points, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if len(points) > 2:  # En az 3 nokta gerekli
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)
    return mask

# Beyin yarımküre bölme ve asimetri analizi - Güvenli versiyonu
class BrainSymmetryProcessor:
    @staticmethod
    def compute_asymmetry_map(image):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            h, w = gray.shape
            
            # İşlenmiş görüntü için sabit bir boyut kullan
            target_size = (512, 512)
            gray_resized = cv2.resize(gray, (target_size[1], target_size[0]))
            
            # Sabit boyutta çalış
            mid_line_resized = target_size[1] // 2
            left_half = gray_resized[:, :mid_line_resized].copy()
            right_half = gray_resized[:, mid_line_resized:].copy()
            
            # Çevirme
            flipped_right = cv2.flip(right_half, 1)
            flipped_left = cv2.flip(left_half, 1)
            
            # Fark hesapla
            left_right_diff = cv2.absdiff(left_half, flipped_right)
            right_left_diff = cv2.absdiff(right_half, flipped_left)
            
            # Asimetri haritası
            asymmetry_map = np.zeros_like(gray_resized)
            asymmetry_map[:, :mid_line_resized] = left_right_diff
            asymmetry_map[:, mid_line_resized:] = cv2.flip(right_left_diff, 1)
            
            # Normalizasyon
            asymmetry_map = cv2.normalize(asymmetry_map, None, 0, 255, cv2.NORM_MINMAX)
            asymmetry_map = cv2.GaussianBlur(asymmetry_map, (5, 5), 0)
            
            # Orijinal boyuta geri dön
            if h != target_size[0] or w != target_size[1]:
                asymmetry_map = cv2.resize(asymmetry_map, (w, h))
            
            return asymmetry_map
            
        except Exception as e:
            print(f"Asimetri haritası oluşturma hatası: {e}")
            # Hata durumunda gri bir görüntü döndür
            return np.zeros_like(image[:,:,0] if len(image.shape)==3 else image)

# Anatomik Simetri Modülü
class AnatomicSymmetryModule(nn.Module):
    def __init__(self, in_channels):
        super(AnatomicSymmetryModule, self).__init__()
        self.conv_left = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_right = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_diff = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_combine = nn.Conv2d(in_channels*3, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        batch_size, c, h, w = x.size()
        mid = w // 2
        
        # Yarımküreleri böl
        left_half = x[:, :, :, :mid]
        right_half = x[:, :, :, mid:]
        
        # Sağ yarımküreyi yatay çevir
        flipped_right = torch.flip(right_half, [3])
        
        # Konvolüsyonlar
        left_feats = self.conv_left(left_half)
        right_feats = self.conv_right(flipped_right)
        
        # Asimetriyi hesapla
        diff = torch.abs(left_feats - right_feats)
        diff_feats = self.conv_diff(diff)
        
        # Asimetri haritası ve orijinal görüntüyü birleştir
        left_padded = F.pad(left_feats, (0, w-mid, 0, 0))
        right_flipped = torch.flip(right_feats, [3])
        right_padded = F.pad(right_flipped, (mid, 0, 0, 0))
        diff_padded = F.pad(diff_feats, (0, w-mid, 0, 0))
        
        combined = torch.cat([x, left_padded, diff_padded], dim=1)
        out = self.relu(self.bn(self.conv_combine(combined)))
        
        return out

# Vasküler Dikkat Modülü
class VascularAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(VascularAttentionModule, self).__init__()
        
        # Çok ölçekli konvolüsyonlar
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=5, padding=2),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # Kanal dikkat mekanizması
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Uzamsal dikkat
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Çok ölçekli özellik çıkarma
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        multi_scale = torch.cat([b1, b2, b3, b4], dim=1)
        
        # Kanal dikkat mekanizması
        avg_out = self.fc(self.avg_pool(multi_scale))
        max_out = self.fc(self.max_pool(multi_scale))
        channel_att = avg_out + max_out
        
        multi_scale = multi_scale * channel_att
        
        # Uzamsal dikkat mekanizması
        avg_out = torch.mean(multi_scale, dim=1, keepdim=True)
        max_out, _ = torch.max(multi_scale, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial_in))
        
        out = multi_scale * spatial_att
        
        return out

# Radyomik Özellik Çıkarma Modülü
class RadiomicFeatureModule(nn.Module):
    def __init__(self, in_channels):
        super(RadiomicFeatureModule, self).__init__()
        
        # Hounsfield birimlerini modelleyen filtreler
        self.intensity_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Doku analizi için Gabor filtre benzeri yapı
        self.texture_conv1 = nn.Conv2d(in_channels, 8, kernel_size=5, padding=2)
        self.texture_conv2 = nn.Conv2d(in_channels, 8, kernel_size=7, padding=3)
        self.texture_conv3 = nn.Conv2d(in_channels, 8, kernel_size=9, padding=4)
        
        # Entegrasyon
        self.combine = nn.Sequential(
            nn.Conv2d(40, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Yoğunluk özellikleri
        int_features = self.intensity_conv(x)
        
        # Doku özellikleri - farklı ölçeklerde
        tex1 = self.texture_conv1(x)
        tex2 = self.texture_conv2(x)
        tex3 = self.texture_conv3(x)
        
        # Özellik birleştirme
        combined = torch.cat([int_features, tex1, tex2, tex3], dim=1)
        
        return self.combine(combined)

# HemisphericComparisonNet - Ana Model
class HemisphericComparisonNet(nn.Module):
    def __init__(self, spatial_dims=2, in_channels=3, out_channels=2):
        super(HemisphericComparisonNet, self).__init__()
        
        # Anatomik Simetri Modülü
        self.symmetry_module = AnatomicSymmetryModule(in_channels)
        
        # İlk katmanlar
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Vasküler Dikkat Modülü
        self.vascular_attention = VascularAttentionModule(64)
        
        # MONAI DenseNet121 omurga
        self.backbone = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=64,
            out_channels=1024,
            pretrained=True
        )
        
        # Radyomik özellik modülü
        self.radiomic_module = RadiomicFeatureModule(in_channels)
        
        # Son sınıflandırma katmanları
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, out_channels)
        )
        
    def forward(self, x):
        # Anatomik simetri özellikleri
        sym_features = self.symmetry_module(x)
        
        # Radyomik özellikler
        radiomic_features = self.radiomic_module(x)
        
        # Özellikleri birleştir
        combined = sym_features + radiomic_features
        
        # İlk konvolüsyon
        x = self.conv1(combined)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Vasküler dikkat uygula
        x = self.vascular_attention(x)
        
        # DenseNet backbone
        x = self.backbone.features(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Sınıflandırma
        x = self.classifier(x)
        
        return x

# Gelişmiş veri seti sınıfı
class AdvancedStrokeDataset(Dataset):
    def __init__(self, normal_dir, stroke_dir, annotations_path=None, transform=None):
        self.transform = transform
        self.annotations = {}
        
        if annotations_path and os.path.exists(annotations_path):
            self.annotations = load_annotations(annotations_path)
        
        # Görüntü yolları
        self.normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.stroke_images = [os.path.join(stroke_dir, f) for f in os.listdir(stroke_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Tüm görüntüler
        self.all_images = self.normal_images + self.stroke_images
        self.labels = [0] * len(self.normal_images) + [1] * len(self.stroke_images)
        
        # Karıştır
        combined = list(zip(self.all_images, self.labels))
        random.shuffle(combined)
        self.all_images, self.labels = zip(*combined)
        self.all_images, self.labels = list(self.all_images), list(self.labels)
        
        print(f"Normal görüntü sayısı: {len(self.normal_images)}")
        print(f"İnme görüntü sayısı: {len(self.stroke_images)}")
        print(f"Toplam görüntü sayısı: {len(self.all_images)}")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        filename = os.path.basename(img_path)
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]
        
        # Asimetri haritası hesapla
        asymmetry_map = BrainSymmetryProcessor.compute_asymmetry_map(image)
        
        # İnme segmentasyon maskelerini oluştur
        h, w = image.shape[:2]
        mask_iskeme = np.zeros((h, w), dtype=np.uint8)
        mask_kanama = np.zeros((h, w), dtype=np.uint8)
        
        # JSON'dan segmentasyon bilgilerini ekle
        if "iskeme" in self.annotations and filename in self.annotations["iskeme"]:
            for region in self.annotations["iskeme"][filename]:
                points = region.get("points", [])
                if points:
                    region_mask = create_mask_from_points(points, image.shape)
                    mask_iskeme = np.logical_or(mask_iskeme, region_mask).astype(np.uint8)
        
        if "kanama" in self.annotations and filename in self.annotations["kanama"]:
            for region in self.annotations["kanama"][filename]:
                points = region.get("points", [])
                if points:
                    region_mask = create_mask_from_points(points, image.shape)
                    mask_kanama = np.logical_or(mask_kanama, region_mask).astype(np.uint8)
        
        # Asimetri haritasını üçüncü kanal olarak ekle
        asymmetry_rgb = np.stack([asymmetry_map]*3, axis=2)
        
        # NumPy dizilerini hazırla
        image = image.astype(np.float32) / 255.0
        asymmetry_rgb = asymmetry_rgb.astype(np.float32) / 255.0
        
        # Görüntüleri tensöre dönüştür
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        asymmetry = torch.from_numpy(asymmetry_rgb).permute(2, 0, 1)  # HWC -> CHW
        
        # Veri dönüşümlerini uygula
        if self.transform:
            image = self.transform(image)
            asymmetry = self.transform(asymmetry)
        
        # Segmentasyon maskelerini tensor'a dönüştür
        mask_iskeme = torch.from_numpy(mask_iskeme).float().unsqueeze(0)
        mask_kanama = torch.from_numpy(mask_kanama).float().unsqueeze(0)
        
        # İnme tipini belirle (0: yok, 1: iskemik, 2: hemorajik, 3: her ikisi)
        stroke_type = 0
        if np.sum(mask_iskeme.numpy()) > 0 and np.sum(mask_kanama.numpy()) == 0:
            stroke_type = 1  # İskemik inme
        elif np.sum(mask_iskeme.numpy()) == 0 and np.sum(mask_kanama.numpy()) > 0:
            stroke_type = 2  # Hemorajik inme
        elif np.sum(mask_iskeme.numpy()) > 0 and np.sum(mask_kanama.numpy()) > 0:
            stroke_type = 3  # Her iki inme tipi
        
        return {
            "image": image,
            "asymmetry": asymmetry,
            "label": torch.tensor(label, dtype=torch.long),
            "mask_iskeme": mask_iskeme,
            "mask_kanama": mask_kanama,
            "stroke_type": torch.tensor(stroke_type, dtype=torch.long),
            "filename": filename
        }

# Eğitim fonksiyonu
def train_model(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with tqdm(train_loader, desc=f"Eğitim Epoch {epoch}/{total_epochs}") as pbar:
        for batch_data in pbar:
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            # Gradyanları sıfırla
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward ve optimizasyon
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Doğruluk hesaplama
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Progress bar güncellemesi
            pbar.set_postfix({"loss": loss.item(), "acc": 100. * correct / total})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels

# Doğrulama fonksiyonu
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Doğrulama") as pbar:
            for batch_data in pbar:
                images = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                filenames = batch_data["filename"]
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_filenames.extend(filenames)
                
                running_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({"loss": loss.item(), "acc": 100. * correct / total})
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return val_loss, val_acc, val_f1, all_preds, all_labels, all_probs, all_filenames

# Test fonksiyonu - External Validation
def test_model(model, test_loader, device, results_dir=RESULTS_DIR):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_filenames = []
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Test") as pbar:
            for batch_data in pbar:
                images = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)
                filenames = batch_data["filename"]
                
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_filenames.extend(filenames)
    
    # Performans metrikleri
    accuracy = 100 * sum([1 for i, j in zip(all_preds, all_labels) if i == j]) / len(all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # ROC eğrisi
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall eğrisi
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    
    # Sınıflandırma raporu
    report = classification_report(all_labels, all_preds, target_names=['İnme Yok', 'İnme Var'])
    
    # Karışıklık matrisi
    cm = confusion_matrix(all_labels, all_preds)
    
    # Sonuçları CSV'ye kaydet
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_label': all_labels,
        'predicted': all_preds,
        'probability': all_probs
    })
    
    results_path = os.path.join(results_dir, f"test_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'report': report,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'predictions': results_df
    }
    
    return results

# Performans görselleştirme fonksiyonu
def visualize_results(train_history, val_history, test_results, save_dir=RESULTS_DIR):
    plt.figure(figsize=(20, 15))
    
    # Kayıp grafiği
    plt.subplot(3, 2, 1)
    plt.plot(train_history['loss'], label='Eğitim')
    plt.plot(val_history['loss'], label='Doğrulama')
    plt.title('Epoch başına Kayıp')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    
    # Doğruluk grafiği
    plt.subplot(3, 2, 2)
    plt.plot(train_history['acc'], label='Eğitim')
    plt.plot(val_history['acc'], label='Doğrulama')
    plt.title('Epoch başına Doğruluk')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk (%)')
    plt.legend()
    
    # F1 skoru grafiği
    plt.subplot(3, 2, 3)
    plt.plot(train_history['f1'], label='Eğitim')
    plt.plot(val_history['f1'], label='Doğrulama')
    plt.title('Epoch başına F1 Skoru')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Skoru')
    plt.legend()
    
    # Test Karışıklık Matrisi
    plt.subplot(3, 2, 4)
    sns.heatmap(test_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['İnme Yok', 'İnme Var'],
                yticklabels=['İnme Yok', 'İnme Var'])
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Karışıklık Matrisi (Test)')
    
    # ROC Eğrisi
    plt.subplot(3, 2, 5)
    plt.plot(test_results['fpr'], test_results['tpr'], color='darkorange', lw=2,
             label=f'ROC eğrisi (AUC = {test_results["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı')
    plt.ylabel('Doğru Pozitif Oranı')
    plt.title('ROC Eğrisi (Test)')
    plt.legend(loc="lower right")
    
    # Precision-Recall Eğrisi
    plt.subplot(3, 2, 6)
    plt.plot(test_results['recall'], test_results['precision'], color='green', lw=2,
             label=f'PR eğrisi (AUC = {test_results["pr_auc"]:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Eğrisi (Test)')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    
    # Kaydet
    plt.savefig(os.path.join(save_dir, f"performance_results_{timestamp}.png"), dpi=300)
    plt.show()

    # Sınıflandırma raporu
    print("\nTest Sonuçları:")
    print(f"Doğruluk: {test_results['accuracy']:.2f}%")
    print(f"F1 Skoru: {test_results['f1_score']:.4f}")
    print(f"ROC AUC: {test_results['roc_auc']:.4f}")
    print(f"PR AUC: {test_results['pr_auc']:.4f}")
    print("\nSınıflandırma Raporu:")
    print(test_results['report'])

# Ana fonksiyon
def main():
    # MONAI dönüşümleri - ToTensor() kullanmıyoruz, veriler zaten tensor formatında
    train_transforms = Compose([
        Resize((512, 512)),
        RandRotate(range_x=15, prob=0.5),
        RandFlip(spatial_axis=0, prob=0.5),
        RandAffine(prob=0.5, translate_range=(0.05, 0.05), scale_range=(0.05, 0.05)),
        Resize((512, 512)),  # Rastgele dönüşümlerden sonra sabit boyuta getir
        RandGaussianNoise(prob=0.2, std=0.01),
        RandAdjustContrast(prob=0.2, gamma=(0.8, 1.2)),
        RandHistogramShift(prob=0.2, num_control_points=10),
        ScaleIntensity(),
    ])
    
    val_transforms = Compose([
        Resize((512, 512)),
        ScaleIntensity(),
    ])
    
    # Veri setlerini yükle
    print("Veri setleri yükleniyor...")
    
    # Eğitim ve test veri setleri
    train_dataset = AdvancedStrokeDataset(
        NORMAL_DIR, STROKE_DIR, 
        annotations_path=ANNOTATIONS_JSON,
        transform=train_transforms
    )
    
    val_dataset = AdvancedStrokeDataset(
        NORMAL_DIR, STROKE_DIR, 
        annotations_path=ANNOTATIONS_JSON,
        transform=val_transforms
    )
    
    test_dataset = AdvancedStrokeDataset(
        TEST_NORMAL_DIR, TEST_STROKE_DIR, 
        annotations_path=ANNOTATIONS_JSON,
        transform=val_transforms
    )
    
    # Eğitim/doğrulama bölünmesi (95%/5%)
    val_ratio = 0.05
    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    
    # İndeksleri oluştur
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Veri alt kümelerini oluştur
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Veri yükleyiciler - pad_list_data_collate kullan
    train_loader = DataLoader(
        train_subset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        collate_fn=pad_list_data_collate,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=pad_list_data_collate,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        collate_fn=pad_list_data_collate,
        drop_last=False
    )
    
    # Sınıf ağırlıklarını doğrudan hesapla (daha hızlı yöntem)
    print("Sınıf ağırlıkları hesaplanıyor...")
    class_counts = np.array([len(train_dataset.normal_images), len(train_dataset.stroke_images)])
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    print(f"Sınıf ağırlıkları: {class_weights}")
    
    # Model oluştur
    print("Model oluşturuluyor...")
    model = HemisphericComparisonNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=2
    ).to(device)
    
    # Kayıp fonksiyonu - to_onehot_y parametresi
    focal_loss = FocalLoss(gamma=2.0, weight=class_weights, to_onehot_y=True)
    dice_loss = DiceLoss(sigmoid=True, smooth_nr=0.01, smooth_dr=0.01)
    
    def combined_loss(outputs, targets):
        # Sigmoid çıktıları
        outputs_sigmoid = torch.sigmoid(outputs)
        
        # Kayıpları hesapla - targets otomatik olarak one-hot'a dönüştürülecek
        fl = focal_loss(outputs, targets)
        
        # DiceLoss için one-hot dönüşümü
        targets_onehot = F.one_hot(targets, num_classes=2).float()
        targets_onehot = targets_onehot.permute(0, 1)
        dl = dice_loss(outputs_sigmoid, targets_onehot)
        
        # Ağırlıklı kombinasyon
        return 0.7 * fl + 0.3 * dl
    
    criterion = combined_loss
    
    # Optimizer - daha düşük öğrenme oranı
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # Eğitim geçmişi
    train_history = {'loss': [], 'acc': [], 'f1': []}
    val_history = {'loss': [], 'acc': [], 'f1': []}
    
    # En iyi doğrulama metriklerini takip et (early stopping yerine)
    best_val_loss = float('inf')
    best_val_f1 = 0
    
    # Eğitim döngüsü
    print("Eğitim başlıyor...")
    epochs = 50
    
    for epoch in range(1, epochs+1):
        # Eğitim
        train_loss, train_acc, train_f1, train_preds, train_labels = train_model(
            model, train_loader, optimizer, criterion, device, epoch, epochs
        )
        
        # Doğrulama
        val_loss, val_acc, val_f1, val_preds, val_labels, val_probs, val_filenames = validate_model(
            model, val_loader, criterion, device
        )
        
        # Scheduler güncelleme
        scheduler.step(val_loss)
        
        # Geçmiş kaydet
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        train_history['f1'].append(train_f1)
        
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)
        val_history['f1'].append(val_f1)
        
        # Sonuçları yazdır
        print(f"Epoch {epoch}/{epochs}")
        print(f"Eğitim: Kayıp={train_loss:.4f}, Doğruluk={train_acc:.2f}%, F1={train_f1:.4f}")
        print(f"Doğrulama: Kayıp={val_loss:.4f}, Doğruluk={val_acc:.2f}%, F1={val_f1:.4f}")
        
        # En iyi modelleri kaydet (early stopping yerine)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_loss_model_{timestamp}.pth"))
            print(f"En iyi doğrulama kaybı modeli kaydedildi: {val_loss:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"En iyi doğrulama F1 modeli kaydedildi: {val_f1:.4f}")
        
        # Son modeli kaydet
        torch.save(model.state_dict(), MODEL_PATH)
    
    # En iyi modeli yükle
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    
    # Test et
    print("Model test ediliyor...")
    test_results = test_model(model, test_loader, device)
    
    # Sonuçları görselleştir
    visualize_results(train_history, val_history, test_results)
    
    # Modeli ve sonuçları kaydet
    print(f"En iyi model '{BEST_MODEL_PATH}' olarak kaydedildi.")
    print(f"Test sonuçları '{RESULTS_DIR}' klasörüne kaydedildi.")
    
    # TEKNOFEST özet raporu
    summary_path = os.path.join(RESULTS_DIR, f"teknofest_summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("TEKNOFEST İnme Tespiti Modeli - Sonuçlar\n")
        f.write("=======================================\n\n")
        f.write("Model: HemisphericComparisonNet (Anatomik Simetri + Vasküler Dikkat)\n")
        f.write(f"Eğitim veri seti boyutu: {len(train_subset)}\n")
        f.write(f"Doğrulama veri seti boyutu: {len(val_subset)}\n")
        f.write(f"Test veri seti boyutu: {len(test_dataset)}\n\n")
        f.write(f"Doğrulama F1 skoru: {max(val_history['f1']):.4f}\n")
        f.write(f"Test F1 skoru: {test_results['f1_score']:.4f}\n")
        f.write(f"Test doğruluğu: {test_results['accuracy']:.2f}%\n")
        f.write(f"Test ROC AUC: {test_results['roc_auc']:.4f}\n\n")
        f.write("Sınıflandırma Raporu (Test):\n")
        f.write(test_results['report'])
    
    print(f"TEKNOFEST özet raporu '{summary_path}' olarak kaydedildi.")

if __name__ == "__main__":
    main()