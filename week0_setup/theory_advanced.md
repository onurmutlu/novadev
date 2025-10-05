# Week 0: İleri Konular & Pratik İpuçları

**NovaDev v1.0 - İleri Seviye Notlar**

> "Teori bilirsen debug yaparsın, pratiği bilirsen hızlı gidersin."

---

## 9️⃣ Reproduksiyon: Bilimin Omurgası

### 🎯 Neden Önemli?

**İddia:** Aynı koşullarda aynı sonuç → Güven

Bilim **tekrarlanabilirlik** üzerine kuruludur. ML'de bu özellikle zor ama kritik.

### 🔒 Seed (Tohum) Sabitleme

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic mode (biraz yavaşlatır)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Kullanım
set_seed(42)
```

**Dikkat:** MPS'de (Apple Silicon) tam determinizm garanti edilemez.

### 📋 Experiment Log Template

```markdown
## Deney 2025-10-06-A

**Amaç:** LR'ın etkisini test et

**Setup:**
- Model: LinearRegression (input=10, output=1)
- Optimizer: Adam
- LR: 0.01
- Batch size: 32
- Epochs: 100
- Seed: 42
- Device: MPS

**Versiyon:**
- Python: 3.13.7
- PyTorch: 2.8.0
- OS: macOS 14.6

**Sonuç:**
- Train Loss: 0.045
- Val Loss: 0.052
- Time: 15.3s

**Gözlem:**
Loss smooth düştü, overfit yok.

**Sonraki adım:**
LR'ı 0.001'e düşür, karşılaştır.
```

### 💾 Checkpoint Kaydetme

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'random_state': torch.get_rng_state(),
    }, path)
    
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['random_state'])
    return checkpoint['epoch'], checkpoint['loss']
```

### 📊 Kural 7: Günlük Deney

> "Bugün 1 deney koşmadıysan, öğrenmedin."

**Neden?**
- Teori okumak ≠ Anlamak
- Kod yazmak ≠ Öğrenmek
- **Deney yapmak = Gerçek öğrenme**

**Pratik:**
- Her gün en az 1 hiperparametre değiştir
- Sonucu not al
- Önceki deneylerle karşılaştır

---

## 🔟 Donanım: MPS/CUDA/CPU Gerçekleri

### 🖥️ Cihaz Karşılaştırması

```
┌────────────────┬──────────┬──────────┬──────────┐
│                │   CPU    │   MPS    │  CUDA    │
├────────────────┼──────────┼──────────┼──────────┤
│ Hız (7B model) │  Yavaş   │  Orta    │  Hızlı   │
│ Uyumluluk      │  %100    │  %95     │  %98     │
│ Kurulum        │  Kolay   │  Kolay   │  Orta    │
│ Bellek         │  Düşük   │  Orta    │  Yüksek  │
│ Maliyet        │  $0      │  Mac     │  GPU $   │
└────────────────┴──────────┴──────────┴──────────┘
```

### 🎚️ Batch Size Etkisi

```python
# Küçük batch (8, 16)
✅ Daha gürültülü gradient
✅ Bazen daha iyi genelleme
✅ Daha az bellek
❌ Yavaş (GPU'yu tam kullanmıyor)

# Büyük batch (128, 256)
✅ Hızlı (GPU parallelizmi)
✅ Stabil gradient
❌ Daha fazla bellek
❌ "Keskin" minimumlara takılabilir
```

**Optimal Strateji:**
1. GPU belleğine sığacak en büyük batch
2. Ama 256'dan büyük nadiren gerekir
3. İkinin katları tercih et (32, 64, 128)

### ⚡ Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in epochs:
    optimizer.zero_grad()
    
    # Forward pass float16'da
    with autocast():
        output = model(input)
        loss = criterion(output, target)
    
    # Backward pass scale edilmiş
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Kazanç:**
- ~2x hız
- Daha az bellek
- Minimal kayıp (genelde fark edilmez)

**Dikkat:** MPS'de henüz tam destek yok (2024 itibarıyla).

### 🎯 Kural 8: Basit Çalış

```
İlk iterasyon:
  ❌ Mixed precision + multi-GPU + compiled model
  ✅ Tek GPU + float32 + basit kod
  
İkinci iterasyon:
  ✅ Şimdi optimize et
```

**Neden?**
- Erken optimizasyon = motivasyon katili
- Önce **çalışır** yap, sonra **hızlı** yap

---

## 1️⃣1️⃣ Sayısal Sağlık (Numerical Stability)

### 🔥 Yaygın Sorunlar

#### Problem 1: NaN (Not a Number)

**Belirtiler:**
```python
Loss: 1.234
Loss: 0.876
Loss: 0.543
Loss: nan  ← Patlama!
```

**Nedenler:**
1. LR çok büyük → Gradient patlaması
2. Bölme sıfıra → Infinity
3. Log(0) veya log(negatif)
4. Sayı taşması (overflow)

**Çözümler:**
```python
# 1. LR küçült
lr = lr / 10

# 2. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Güvenli operasyonlar
log_prob = torch.log(prob + 1e-8)  # Epsilon ekle

# 4. Initialization
# Küçük ağırlıklar başlat
```

#### Problem 2: Gradient Explosion

**Belirtiler:**
```
Gradient norm: 10.5
Gradient norm: 23.8
Gradient norm: 156.9
Gradient norm: 8921.4  ← Patlama!
```

**Çözüm: Gradient Clipping**
```python
# Norm-based clipping
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # Gradient'leri 1.0'a sınırla
)

# Value-based clipping
torch.nn.utils.clip_grad_value_(
    model.parameters(),
    clip_value=0.5
)
```

#### Problem 3: Gradient Vanishing

**Belirtiler:**
```
Layer 1 gradient: 0.5
Layer 2 gradient: 0.05
Layer 3 gradient: 0.0005
Layer 4 gradient: 0.0000001  ← Kayboluyor!
```

**Neden:**
Derin ağlarda türevler çarpılırken küçülür.

**Çözümler:**
1. **ReLU activation** (sigmoid yerine)
2. **Batch Normalization**
3. **Residual connections** (ResNet'te)
4. Daha iyi **initialization** (Xavier, He)

### 🎯 Kural 9: İlk Panik Anında

```
1. LR'ı yarıya indir
2. Tekrar dene
3. %80 oranla düzelir!
```

**Teşhis Akışı:**
```
NaN gördün mü?
  ↓
LR'ı /2 yap
  ↓
Düzeldi mi?
  ├─ Evet → Devam et
  └─ Hayır → Gradient clip ekle
              ↓
            Düzeldi mi?
              ├─ Evet → Devam et
              └─ Hayır → Model/veri kontrol et
```

---

## 1️⃣2️⃣ Gerçek Hayata Bağ

### 🎯 Neden Bu Kadar Teori?

**Senaryo 1: Model çalışmıyor**
```
Acemi:
  "Kod çalışmıyor, Stack Overflow'a bakalım"
  → 3 saat kopyala-yapıştır
  → Hala çalışmıyor

Expert:
  "Loss eğrisine bakalım..."
  → LR çok büyük (5 dakika)
  → loss.backward() sonrası zero_grad() yok (2 dakika)
  → Düzeldi! (7 dakika)
```

**Fark:** Teori → hızlı teşhis

### 🔍 Hızlı Teşhis Check-list

```markdown
## Model Çalışmıyor - 6 Adım Protokol

### 1. LR Kontrolü
- [ ] LR çok mu yüksek? (loss zıplıyor mu?)
- [ ] LR çok mu düşük? (loss hareket etmiyor mu?)

### 2. Veri Sızıntısı
- [ ] Train/val/test doğru mu ayrıldı?
- [ ] Normalizasyon train'den mi öğrenildi?
- [ ] Shuffle yapıldı mı?

### 3. Shape/Device Kontrolü
- [ ] Tüm tensor'ler aynı device'da mı?
- [ ] Shape'ler beklediğin gibi mi?
- [ ] Dtype uyumlu mu? (float32 vs int64)

### 4. Gradient Akışı
- [ ] zero_grad() her iterasyonda mı?
- [ ] backward() çağrılıyor mu?
- [ ] requires_grad=True parametrelerde mi?

### 5. Loss Fonksiyonu
- [ ] Doğru loss seçildi mi? (MSE ≠ CE)
- [ ] Loss reduction doğru mu? (mean/sum)
- [ ] Target shape doğru mu?

### 6. Seed & Randomness
- [ ] Seed sabitlendi mi?
- [ ] Sonuçlar reprodukılabilir mi?
```

### 🎓 Gerçek Hikayeler

#### Hikaye 1: "İki Haftayı Boşa Harcadım"
```
Öğrenci: Val loss hiç iyileşmiyor!
         2 hafta denedim, her şeyi denedim.

Mentor:  Veri nasıl ayrıldı?
Öğrenci: Önce normalize ettim, sonra ayırdım.
Mentor:  İşte sorun! Test bilgisi train'e sızdı.

Çözüm:   10 dakika (veri pipeline'ı düzelt)
Ders:    Teori bilmek = Zaman kazanmak
```

#### Hikaye 2: "Model Patladı"
```
Mühendis: Production'da model NaN döndürüyor!
          Eğitimde sorun yoktu.

Debug:    - Prod'da veri dağılımı değişmiş
          - Outlier'lar var
          - Log(0) patlaması

Çözüm:    - Input clipping ekle
          - Robust loss kullan (Huber)
          - Monitoring kur

Ders:     Train ≠ Prod, savunmalı kod yaz
```

### 💡 Professional vs Amateur

```
┌──────────────────┬──────────────┬──────────────┐
│                  │   Amateur    │  Professional│
├──────────────────┼──────────────┼──────────────┤
│ Problem          │ Panik yapar  │ Checklist    │
│ Debug            │ Random try   │ Systematic   │
│ Experiment       │ Kaydı yok    │ Log tutar    │
│ Code             │ Karmaşık     │ Basit        │
│ Reproduksiyon    │ Şans         │ Garanti      │
│ Error            │ "Çalışmıyor" │ "LR büyük"   │
└──────────────────┴──────────────┴──────────────┘
```

---

## 📚 Edebiyat Özeti

**Makine Öğrenmesi = Fonksiyon tasarlama sanatı**

Süreç:
1. **Veri**ni ciddiye al (garbage in = garbage out)
2. **Hedef**'i net kur (loss fonksiyonu)
3. **Model**'i başlat (parametreler rastgele)
4. **Gradient** ile yön bul (pusıla)
5. **LR** ile adım at (hız kontrol)
6. **Val** ile kontrol et (vicdan)
7. **Regularize** et (ezber değil öğrenme)
8. **Tekrarla** (epoch döngüsü)

**Sonuç:** Sihir değil, **güvenilir işlev**
- Veri değiştikçe uyarlanabilir
- Hatası ölçülebilir
- Tekrarlanabilir

---

## 🎨 Ek Analojiler

### 1. Gradient = Damla Analojisi

```
        Dağ Zirvesi
           ╱╲
         ╱    ╲
       ╱        ╲
     ╱     ●      ╲    ← Su damlası
   ╱       ↓        ╲
 ╱         ↓          ╲
───────────●────────────  ← Vadi (minimum)
```

Su damlası:
- En dik eğimden akar (gradient)
- Küçük adımlar atar
- Yerel minimum'lara takılabilir
- Momentum ile atlar

**Model eğitimi:** Binlerce damla aynı anda akıyor (mini-batch)

### 2. Loss Yüzeyi = Topoğrafya

```
3D Loss Landscape:

        ╱╲    ╱╲
       ╱  ╲  ╱  ╲
      ╱ ●  ╲╱    ╲   ← Yerel minimum
     ╱            ╲
    ╱              ╲
   ╱      ◆         ╲  ← Global minimum
  ──────────────────────

● = Kötü (yerel)
◆ = İyi (global)
```

**Amaç:** ◆'ye ulaşmak
**Problem:** ●'de takılabilirsin
**Çözüm:** Momentum, SGD'nin gürültüsü, farklı init

### 3. Model = Radyo (Geliştirilmiş)

```
┌──────────────────────┐
│  [∿] [○] [═] [▫]   │  ← Düğmeler (parametreler)
│                      │
│   🔊  ♫ ♪ ♫          │  ← Çıktı (tahmin)
│                      │
│   S/N Ratio: 85%    │  ← Loss (gürültü/sinyal)
└──────────────────────┘

Başlangıç: Her düğme rastgele → Gürültü
Eğitim:    Düğmeleri ayarla → Müzik netleşir
Gradient:  "Bu düğmeyi sola çevir" sinyali
LR:        Düğmeyi ne kadar çevireceğin
```

### 4. Overfit = Ezber Yapan Öğrenci

```
Ezberci Öğrenci:
  Sınav soruları: %100 ✓
  Farklı sorular: %40 ✗
  → Anlamadı, ezberledi

Öğrenen Öğrenci:
  Sınav soruları: %85 ✓
  Farklı sorular: %80 ✓
  → Kavradı, genelledi
```

**Model:** Overfit = Ezber
**Çözüm:** Regularization = "Anlama" zorla

---

## 🧪 İleri Alıştırmalar

### Alıştırma 4: Loss Landscape Hayal Et

**Senaryo:** 2D loss yüzeyi hayal et.

```
Loss(w1, w2)
      ↑
      │     ╱╲
      │   ╱    ╲
      │ ╱        ╲
      └─────────────→ w1
         w2
```

**Sorular:**
1. Gradient hangi yönü gösterir?
2. LR büyükse ne olur?
3. Momentum ne işe yarar?

### Alıştırma 5: Batch Size Deneyi

**Kod:**
```python
model_small = train(model, batch_size=8)
model_large = train(model, batch_size=256)
```

**Sorular:**
1. Hangisi daha hızlı epoch bitirir?
2. Hangisi daha az bellek kullanır?
3. Hangisi daha iyi genelleme yapabilir?

### Alıştırma 6: Debug Senaryosu

**Problem:**
```
Epoch 1: Loss = 2.5
Epoch 2: Loss = nan
```

**Sorular:**
1. İlk şüphen ne?
2. Hangi satırı kontrol edersin?
3. Hızlı fix'in ne?

---

### 📝 Cevaplar

**Alıştırma 4:**
1. En dik iniş yönü (negatif gradient)
2. Minimum'u aşar, zıplar (diverge)
3. Momentum vadileri hızlı aşar, lokal min'lerden kaçar

**Alıştırma 5:**
1. batch_size=256 daha hızlı (GPU parallelizmi)
2. batch_size=8 daha az bellek
3. batch_size=8 bazen daha iyi (gürültülü gradient → regularization)

**Alıştırma 6:**
1. İlk şüphe: **LR çok büyük**
2. Kontrol: `print(loss)` ve `print(model.parameters())`
3. Fix: `lr = lr / 10`

---

## 🎓 Kapanış: Sen Neredesin?

### Self-Assessment

**Başlangıç (Week 0 öncesi):**
```
[ ] Tensör nedir bilmiyordum
[ ] Gradient manuel hesaplamıştım ama autograd yabancı
[ ] Loss fonksiyonu seçimi şans işiydi
[ ] Overfit/underfit farkını bilmiyordum
```

**Hedef (Week 0 sonrası):**
```
[✓] Tensör operasyonlarını anlıyorum
[✓] Autograd akışını görselleştirebiliyorum
[✓] Loss/optimizer/LR ilişkisini kavradım
[✓] Debug yapabilirim (check-list ile)
[✓] Deney tasarlayıp kaydedebilirim
```

**İleri (Week 4+):**
```
[ ] Kendi modelimi sıfırdan yazabilirim
[ ] Production pipeline kurabilirim
[ ] Literatür okuyup uygulayabilirim
[ ] Yeni problemlere transfer edebilirim
```

---

## 📖 Önerilen Kaynaklar

### Temel Seviye
- **3Blue1Brown**: Neural Networks (YouTube series)
- **Fast.ai**: Practical Deep Learning for Coders
- **PyTorch Tutorials**: Official docs

### Orta Seviye
- **Deep Learning Book** (Goodfellow et al.) - Bölüm 1-5
- **Dive into Deep Learning** (d2l.ai)
- **CS231n** (Stanford) - Lecture notes

### İleri Seviye
- **Papers with Code**: Son araştırmalar
- **Distill.pub**: Görsel açıklamalar
- **ArXiv Sanity**: Paper takibi

---

## 🚀 Week 1'e Geçiş

**Şu ana kadar:**
- ✅ Teoriyi oturtttuk
- ✅ Kavramları sindirik
- ✅ Sezgi geliştirdik

**Şimdi:**
- 🎯 **Linear Regression** ile pratiğe geç
- 💻 Kodu elle yaz (manuel GD)
- 🧪 Deney yap (LR, batch size, vb.)
- 📊 Sonuçları kaydet

**Motto:** "Theory without practice is lame, practice without theory is blind."

---

## ✅ Final Checklist

Aşağıdaki soruları **kendinize** sorun:

### Kavramsal Anla
- [ ] Model = f_θ(x) ne demek açıklayabiliyorum
- [ ] Gradient nedir, niye önemlidir biliyorum
- [ ] Overfit vs underfit farkını görselleştirebiliyorum
- [ ] LR'ın etkisini tahmin edebiliyorum

### Pratik Beceri
- [ ] PyTorch tensor oluşturabilirim
- [ ] Forward-backward-update döngüsünü yazabilirim
- [ ] Loss eğrisini okuyabilirim
- [ ] Basit hataları debug edebilirim

### Mental Model
- [ ] "Neden?" sorusunu cevaplayabiliyorum
- [ ] Hiperparametre seçimlerini savunabiliyorum
- [ ] Sorun gördüğümde check-list uygulayabiliyorum

**Hepsi ✅ ise:** Week 1 linear regression seni bekliyor! 🎊

---

## 📌 Son Söz

> "Bir öğretmen tahtada anlattı, sen anlamadın.  
> İki YouTube video izledin, yine anlamadın.  
> Üç kez kendi ellerin ile kod yazdın, **anladın**."

**Week 0 bittiğinde:**
- Kafanda net bir **zihinsel model** olmalı
- **Neden?** sorularının cevabını bilmelisin
- İlk hatayı **teşhis edebilmelisin**

**Başarı ölçütü:** Week 1'de kod yazarken "Aha! İşte o yüzden!" diyebilmen.

---

**Hazır mısın?**

```bash
cd /Users/onur/code/novadev-protocol
source .venv/bin/activate
python week1_tensors/linreg_manual.py
```

🚀 **Let's build!**
