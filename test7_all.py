import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import csv
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Model tanımı
class StrokeDetectionModel(nn.Module):
    def __init__(self):
        super(StrokeDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 32 * 32)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_confusion_matrix_plot(cm, output_path):
    """Confusion matrix görselleştirmesi oluşturur"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'İnme Var'],
                yticklabels=['Normal', 'İnme Var'])
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek Değer')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_accuracy_chart(correct_counts, output_path):
    """Doğru/yanlış tahmin sayılarını gösteren çubuk grafik oluşturur"""
    labels = ['Doğru Tahmin', 'Yanlış Tahmin']
    counts = [correct_counts[True], correct_counts[False]]
    total = sum(counts)
    percentages = [count/total*100 for count in counts]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, color=['green', 'red'])
    
    # Çubukların üzerine sayıları ve yüzdeleri ekle
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height} (%{percentages[i]:.1f})', ha='center', va='bottom')
    
    plt.title('Tahmin Performansı')
    plt.ylabel('Görüntü Sayısı')
    plt.ylim(0, max(counts) * 1.1)  # Biraz boşluk bırak
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_class_distribution_chart(class_counts, output_path):
    """Veri setindeki sınıf dağılımını gösteren pasta grafik oluşturur"""
    labels = ['Normal', 'İnme Var']
    sizes = [class_counts[0], class_counts[1]]
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
    plt.axis('equal')
    plt.title('Veri Seti Sınıf Dağılımı')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Yolları ayarlama
    TEST_FOLDER = "/home/kemal/Downloads/DATASET/Sağlık Bakanlığı/YarısmaVeriSeti_1.Oturum/Test Veri Seti_1/PNG"
    MODEL_PATH = 'test7_model.pth'
    
    # Çıktı dosyaları için zaman damgası oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"model_evaluation_{timestamp}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"test_results_{timestamp}.csv")
    OUTPUT_SUMMARY = os.path.join(OUTPUT_DIR, f"test_summary_{timestamp}.txt")
    OUTPUT_CM_PLOT = os.path.join(OUTPUT_DIR, f"confusion_matrix_{timestamp}.png")
    OUTPUT_ACC_PLOT = os.path.join(OUTPUT_DIR, f"accuracy_chart_{timestamp}.png")
    OUTPUT_DIST_PLOT = os.path.join(OUTPUT_DIR, f"class_distribution_{timestamp}.png")
    
    # Ground truth verisi
    ground_truth = {
        '10189': 0, '10200': 0, '10231': 0, '10267': 0, '10300': 0,
        '10301': 0, '10319': 0, '10360': 0, '10467': 0, '10493': 0,
        '10494': 0, '10641': 0, '10693': 0, '10833': 0, '10910': 0,
        '10911': 0, '10917': 0, '10923': 0, '11023': 0, '11072': 0,
        '11308': 0, '11319': 0, '11465': 0, '11482': 0, '11520': 0,
        '11530': 0, '11564': 0, '11612': 0, '11619': 0, '11651': 0,
        '11800': 0, '11925': 0, '11964': 0, '11966': 0, '12018': 0,
        '12079': 0, '12199': 0, '12289': 0, '12344': 0, '12383': 0,
        '12487': 0, '12542': 0, '12603': 0, '12616': 0, '12743': 0,
        '12933': 0, '13021': 0, '13051': 0, '13137': 0, '13179': 0,
        '13345': 0, '13459': 0, '13477': 0, '13497': 0, '13554': 0,
        '13579': 0, '13597': 0, '13600': 0, '13680': 0, '13708': 0,
        '13713': 0, '13731': 0, '13747': 0, '13769': 0, '13770': 0,
        '13896': 0, '14153': 0, '14179': 0, '14198': 0, '14317': 0,
        '14321': 0, '14503': 0, '14522': 0, '14561': 0, '14600': 0,
        '14606': 0, '14621': 0, '14638': 0, '14687': 0, '14798': 0,
        '14892': 0, '14941': 0, '14970': 0, '15028': 0, '15038': 0,
        '15190': 0, '15245': 0, '15268': 0, '15279': 0, '15300': 0,
        '15405': 0, '15517': 0, '15532': 0, '15550': 0, '15614': 0,
        '15618': 0, '15655': 0, '15785': 0, '15868': 0, '15957': 0,
        '15967': 0, '15982': 0, '16006': 0, '16023': 0, '16039': 0,
        '16048': 0, '16057': 0, '16061': 0, '16167': 0, '16257': 0,
        '16263': 0, '16340': 0, '16341': 0, '16358': 0, '16367': 0,
        '16411': 0, '16473': 0, '16493': 0, '16499': 0, '16524': 0,
        '16569': 0, '16728': 0, '16749': 0, '16760': 0, '16798': 0,
        '16846': 0, '16902': 0, '16921': 0, '16969': 0, '17002': 0,
        '10007': 1, '10029': 1, '10437': 1, '10785': 1, '11133': 1,
        '11318': 1, '11336': 1, '11351': 1, '11479': 1, '11963': 1,
        '12009': 1, '12488': 1, '12639': 1, '12645': 1, '12800': 1,
        '12918': 1, '12996': 1, '12997': 1, '13063': 1, '13241': 1,
        '13887': 1, '14031': 1, '14219': 1, '14270': 1, '14343': 1,
        '14549': 1, '14839': 1, '15325': 1, '15708': 1, '16116': 1,
        '16387': 1, '16426': 1, '16953': 1, '17014': 1, '17023': 1,
        '10241': 1, '10461': 1, '10958': 1, '11074': 1, '11104': 1,
        '11114': 1, '11263': 1, '11563': 1, '11669': 1, '11864': 1,
        '11965': 1, '12593': 1, '13065': 1, '13087': 1, '13239': 1,
        '13446': 1, '13447': 1, '13729': 1, '13893': 1, '14101': 1,
        '14183': 1, '14438': 1, '14783': 1, '15363': 1, '15464': 1,
        '15562': 1, '15564': 1, '15622': 1, '15876': 1, '15948': 1,
        '16026': 1, '16478': 1, '16607': 1, '16643': 1, '17017': 1
    }
    
    # Sınıf dağılımını hesapla
    class_counts = {0: 0, 1: 0}
    for label in ground_truth.values():
        class_counts[label] += 1
    
    # Başlangıç zamanını kaydet
    start_time = time.time()
    
    # Cihazı belirle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Modeli yükle
    model = StrokeDetectionModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model başarıyla yüklendi.")
    
    # Dönüşümleri tanımla
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Sonuçları saklamak için liste
    results = []
    
    # Confusion matrix ve istatistikler için değişkenler
    confusion_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    cm_array = np.zeros((2, 2), dtype=int)  # 2x2 confusion matrix (plot için)
    correct_counts = {True: 0, False: 0}
    total_processed = 0
    processed_ids = set()
    
    # Görüntüleri işle
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    all_files = [f for f in os.listdir(TEST_FOLDER) if f.endswith(image_extensions)]
    
    print(f"Toplam {len(all_files)} görüntü dosyası bulundu.")
    print(f"Ground truth'ta {len(ground_truth)} ID bulunuyor.")
    
    # tqdm ile ilerleme çubuğu göster
    for filename in tqdm(all_files, desc="Görüntüler işleniyor"):
        # Dosya adından ID'yi çıkar (uzantıyı kaldır)
        img_id = filename.split('.')[0]
        
        # ID ground truth'ta varsa işle
        if img_id in ground_truth:
            total_processed += 1
            processed_ids.add(img_id)
            img_path = os.path.join(TEST_FOLDER, filename)
            
            try:
                # Görüntüyü yükle ve dönüştür
                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)
                
                # Tahmin yap
                with torch.no_grad():
                    output = model(image)
                    probabilities = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    prediction = predicted.item()
                    confidence_value = confidence.item()
                
                true_label = ground_truth[img_id]
                is_correct = prediction == true_label
                correct_counts[is_correct] += 1
                
                # Etiketin metinsel karşılıkları
                prediction_label = "İnme Var" if prediction == 1 else "Normal"
                true_label_text = "İnme Var" if true_label == 1 else "Normal"
                
                # Confusion matrix güncelle
                if prediction == 1 and true_label == 1:
                    confusion_matrix["TP"] += 1
                    cm_array[1, 1] += 1
                elif prediction == 0 and true_label == 0:
                    confusion_matrix["TN"] += 1
                    cm_array[0, 0] += 1
                elif prediction == 1 and true_label == 0:
                    confusion_matrix["FP"] += 1
                    cm_array[0, 1] += 1
                elif prediction == 0 and true_label == 1:
                    confusion_matrix["FN"] += 1
                    cm_array[1, 0] += 1
                
                # Sonucu sakla
                results.append({
                    'Image_Id': img_id,
                    'Prediction': prediction,
                    'Prediction_Label': prediction_label,
                    'True_Label': true_label,
                    'True_Label_Text': true_label_text,
                    'Confidence': f'{confidence_value:.2%}',  # Yüzde formatına dönüştür
                    'Correct': is_correct,
                    'Correct_Text': "DOĞRU" if is_correct else "YANLIŞ"
                })
                
            except Exception as e:
                print(f"\nHata: {img_id} işlenirken sorun oluştu: {e}")
    
    # İşleme süresini hesapla
    processing_time = time.time() - start_time
    
    # Kayıp ID'leri kontrol et
    all_ground_truth_ids = set(ground_truth.keys())
    missing_ids = all_ground_truth_ids - processed_ids
    
    # Metrikleri hesapla
    total = confusion_matrix["TP"] + confusion_matrix["TN"] + confusion_matrix["FP"] + confusion_matrix["FN"]
    accuracy = (confusion_matrix["TP"] + confusion_matrix["TN"]) / total if total > 0 else 0
    precision = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FP"]) if (confusion_matrix["TP"] + confusion_matrix["FP"]) > 0 else 0
    recall = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"]) if (confusion_matrix["TP"] + confusion_matrix["FN"]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = confusion_matrix["TN"] / (confusion_matrix["TN"] + confusion_matrix["FP"]) if (confusion_matrix["TN"] + confusion_matrix["FP"]) > 0 else 0
    
    # Yüzdelik değerleri hesapla
    accuracy_pct = accuracy * 100
    precision_pct = precision * 100
    recall_pct = recall * 100
    f1_pct = f1 * 100
    specificity_pct = specificity * 100
    
    # Sonuçları detaylı CSV'ye kaydet
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    # Görselleştirmeler oluştur
    create_confusion_matrix_plot(cm_array, OUTPUT_CM_PLOT)
    create_accuracy_chart(correct_counts, OUTPUT_ACC_PLOT)
    create_class_distribution_chart(class_counts, OUTPUT_DIST_PLOT)
    
    # Özet raporu dosyaya yaz
    with open(OUTPUT_SUMMARY, 'w', encoding='utf-8') as summary_file:
        summary_file.write("===== İNME TESPİT MODELİ DEĞERLENDİRME RAPORU =====\n")
        summary_file.write(f"Tarih/Saat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_file.write(f"Model: {MODEL_PATH}\n")
        summary_file.write(f"Test Klasörü: {TEST_FOLDER}\n")
        summary_file.write(f"Kullanılan Cihaz: {device}\n")
        summary_file.write(f"İşlem Süresi: {processing_time:.2f} saniye\n\n")
        
        summary_file.write("--- VERİ SETİ İSTATİSTİKLERİ ---\n")
        summary_file.write(f"Ground Truth'taki Toplam ID Sayısı: {len(ground_truth)}\n")
        summary_file.write(f"İşlenen Toplam Görüntü Sayısı: {total_processed}\n")
        normal_pct = (class_counts[0] / len(ground_truth)) * 100
        inme_pct = (class_counts[1] / len(ground_truth)) * 100
        summary_file.write(f"Normal (Sınıf 0) Görüntü Sayısı: {class_counts[0]} (%{normal_pct:.1f})\n")
        summary_file.write(f"İnme Var (Sınıf 1) Görüntü Sayısı: {class_counts[1]} (%{inme_pct:.1f})\n")
        
        if missing_ids:
            missing_pct = (len(missing_ids) / len(ground_truth)) * 100
            summary_file.write(f"Bulunamayan Görüntü Sayısı: {len(missing_ids)} (%{missing_pct:.1f})\n")
            summary_file.write(f"Bulunamayan ID'ler: {', '.join(sorted(list(missing_ids)))}\n\n")
        else:
            summary_file.write("Tüm ID'ler işlendi, eksik yok.\n\n")
        
        summary_file.write("--- PERFORMANS METRİKLERİ ---\n")
        summary_file.write(f"Doğruluk (Accuracy): %{accuracy_pct:.2f}\n")
        summary_file.write(f"Kesinlik (Precision): %{precision_pct:.2f}\n")
        summary_file.write(f"Duyarlılık (Recall/Sensitivity): %{recall_pct:.2f}\n")
        summary_file.write(f"Özgüllük (Specificity): %{specificity_pct:.2f}\n")
        summary_file.write(f"F1 Skoru: %{f1_pct:.2f}\n\n")
        
        summary_file.write("--- CONFUSION MATRIX ---\n")
        summary_file.write("                 | Tahmin: Normal | Tahmin: İnme Var |\n")
        summary_file.write(f"Gerçek: Normal   | {confusion_matrix['TN']:14d} | {confusion_matrix['FP']:16d} |\n")
        summary_file.write(f"Gerçek: İnme Var | {confusion_matrix['FN']:14d} | {confusion_matrix['TP']:16d} |\n\n")
        
        summary_file.write("--- DOĞRU/YANLIŞ SAYILARI ---\n")
        correct_pct = (correct_counts[True] / total_processed) * 100
        incorrect_pct = (correct_counts[False] / total_processed) * 100
        summary_file.write(f"Doğru Tahmin Sayısı: {correct_counts[True]} (%{correct_pct:.1f})\n")
        summary_file.write(f"Yanlış Tahmin Sayısı: {correct_counts[False]} (%{incorrect_pct:.1f})\n")
    
    # Konsola özet çıktıyı yazdır
    print("\n" + "="*50)
    print("İNME TESPİT MODELİ DEĞERLENDİRME SONUÇLARI")
    print("="*50)
    print(f"İşlenen görüntü sayısı: {total_processed}")
    print(f"İşlem süresi: {processing_time:.2f} saniye")
    print("\nPerformans Metrikleri:")
    print(f"  Doğruluk (Accuracy): %{accuracy_pct:.2f}")
    print(f"  Kesinlik (Precision): %{precision_pct:.2f}")
    print(f"  Duyarlılık (Recall): %{recall_pct:.2f}")
    print(f"  F1 Skoru: %{f1_pct:.2f}")
    print(f"  Özgüllük (Specificity): %{specificity_pct:.2f}")
    
    print("\nConfusion Matrix:")
    print(f"  True Positive (TP): {confusion_matrix['TP']}")
    print(f"  True Negative (TN): {confusion_matrix['TN']}")
    print(f"  False Positive (FP): {confusion_matrix['FP']}")
    print(f"  False Negative (FN): {confusion_matrix['FN']}")
    
    doğru_yüzde = (correct_counts[True] / total_processed) * 100
    print(f"\nDoğru Tahmin: {correct_counts[True]} (%{doğru_yüzde:.2f})")
    yanlış_yüzde = (correct_counts[False] / total_processed) * 100
    print(f"Yanlış Tahmin: {correct_counts[False]} (%{yanlış_yüzde:.2f})")
    
    print("\nSonuçlar ve Grafikler Şuraya Kaydedildi:")
    print(f"  Detaylı CSV: {OUTPUT_CSV}")
    print(f"  Özet Rapor: {OUTPUT_SUMMARY}")
    print(f"  Confusion Matrix Grafiği: {OUTPUT_CM_PLOT}")
    print(f"  Doğruluk Grafiği: {OUTPUT_ACC_PLOT}")
    print(f"  Sınıf Dağılım Grafiği: {OUTPUT_DIST_PLOT}")
    
    if missing_ids:
        missing_pct = (len(missing_ids) / len(ground_truth)) * 100
        print(f"\nUyarı: {len(missing_ids)} ID için görüntü bulunamadı. (%{missing_pct:.1f})")
        print(f"İlk 5 eksik ID: {', '.join(list(missing_ids)[:5])}")
        if len(missing_ids) > 5:
            print("Tüm eksik ID'ler özet raporda listelenmiştir.")

if __name__ == "__main__":
    main()