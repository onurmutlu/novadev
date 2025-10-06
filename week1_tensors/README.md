# Week 1: Linear Regression — Teoriden Pratiğe

**NovaDev v1.0 - Pratik Sprint**

> "Teori bitti. Şimdi kod zamanı. 45 dakikada baştan sona, 5 günde kusursuz."

---

## 🎯 Hedef

**Linear Regression**'u sıfırdan yazıp, eğitim sürecini anlamak:
- ✅ Manuel gradient descent (autograd'ı çıplak görmek)
- ✅ nn.Module ile temiz kod
- ✅ Train/Val split + Early stopping
- ✅ L2 regularization
- ✅ **Val MSE < 0.5**

**Teori Bağlantısı:** Week 0'da öğrendiklerin artık KOD!

---

## 🔥 (A) BUGÜN — 45 Dakikalık Hızlı Pratik

**Amaç:** Linear regression'u baştan sona koş, metrik kaydet, mini rapor çıkar.

### ⏱️ Timeline

#### 0-5 dk: Setup
```bash
# Venv aktif et
source .venv/bin/activate
cd /Users/onur/code/novadev-protocol

# Week 1 klasörüne geç
cd week1_tensors
```

#### 5-15 dk: Data & Manuel GD
```bash
# Sentetik veri üret
python data_synth.py
# Output: data/week1_linreg.pt

# Manuel gradient descent
python linreg_manual.py
# Output: Manuel GD ile MSE hesabı
```

**Ne öğreniyorsun:**
- Sentetik veri oluşturma (y = wx + b + noise)
- Forward pass (ŷ = Xw + b)
- Loss hesabı (MSE)
- Backward pass (autograd)
- Manuel parametre güncelleme (w ← w - η∇L)

#### 15-30 dk: nn.Module + Training Loop
```bash
# nn.Module ile temiz kod
python linreg_module.py

# Train/Val split + Early stopping
python train.py
# Output: outputs/model.pt, outputs/metrics.json
```

**Ne öğreniyorsun:**
- nn.Module yapısı
- DataLoader kullanımı
- Train/Val split
- Early stopping implementasyonu
- L2 regularization (weight_decay)

#### 30-40 dk: Test & Lint
```bash
# Testleri koş
pytest -q tests/test_linreg.py

# Linter kontrol
ruff check week1_tensors/
```

**Ne öğreniyorsun:**
- Pytest ile model testi
- Convergence testi
- L2 regularization etkisi testi

#### 40-45 dk: Mini Rapor
```bash
# Kısa rapor oluştur
cat > week1_summary.md << EOF
# Week 1 - Hızlı Sprint Özeti

## Konfigürasyon
- LR: 5e-3
- L2: 1e-3
- Batch: 32
- Patience: 5

## Sonuçlar
- Train MSE: 0.12
- Val MSE: 0.18
- Epochs: 25 (early stopped)

## Gözlem
- Early stopping çalıştı (epoch 25'te)
- L2 olmadan overfit başlıyordu
- Val MSE < 0.5 ✓
EOF

# Commit
git add .
git commit -m "day1: linreg end-to-end, val MSE=0.18"
```

### ✅ Definition of Done (Bugün)

```
□ data/week1_linreg.pt oluşturuldu
□ Val MSE < 0.5
□ pytest yeşil
□ week1_summary.md var
□ Git commit yapıldı
```

---

## 🚀 (B) Week 1 — 5 Günlük Sprint (2-3 saat/gün)

### 📅 Gün 1: E2E Temel + Metrik Günlüğü

**Hedef:** Tüm pipeline'ı baştan sona koş, kayıt sistemi kur

**Görevler:**
```bash
# 1. Sentetik veri oluştur
python data_synth.py

# 2. Manuel GD
python linreg_manual.py

# 3. nn.Module
python linreg_module.py

# 4. Full training
python train.py
```

**Kayıt Sistemi:**
```bash
# outputs/exp_log.csv oluştur
echo "exp_id,lr,l2,batch,patience,train_mse,val_mse,epochs,stopped_early" > outputs/exp_log.csv
```

**Kabul Kriteri:**
- ✅ Val MSE < 0.5
- ✅ Testler yeşil
- ✅ exp_log.csv ilk satırı dolu

**Çıktı:**
```bash
git commit -m "day1: E2E pipeline + logging system"
```

---

### 📅 Gün 2: Early Stopping + LR Schedule

**Hedef:** Regularization stratejilerini ekle ve etkilerini ölç

**Görevler:**

1. **Early Stopping Ekle** (train.py'da)
```python
best_val_loss = float('inf')
patience_counter = 0
patience = 3  # ← Ekle

for epoch in range(epochs):
    # ... train loop
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint()
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

2. **LR Schedule Ekle**
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# Training loop'ta
scheduler.step(val_loss)
```

3. **3 Deney Koş**
```bash
# Experiment 1: LR=1e-2
python train.py --lr 1e-2 --l2 1e-3 --exp-id exp001

# Experiment 2: LR=5e-3
python train.py --lr 5e-3 --l2 1e-3 --exp-id exp002

# Experiment 3: LR=1e-3
python train.py --lr 1e-3 --l2 1e-3 --exp-id exp003
```

**Kabul Kriteri:**
- ✅ Early stopping tetikliyor
- ✅ LR schedule çalışıyor (log'da görünüyor)
- ✅ exp_log.csv'de 3 satır var
- ✅ En iyi config belirlendi

**Çıktı:**
```bash
git commit -m "day2: early stopping + LR schedule, best: lr=5e-3"
```

---

### 📅 Gün 3: Ölçekleme Ablation'ı

**Hedef:** Feature scaling'in etkisini ölç

**Görevler:**

1. **StandardScaler Ekle**
```python
from sklearn.preprocessing import StandardScaler

# Train'de öğren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Val/Test'e uygula (FIT değil!)
X_val_scaled = scaler.transform(X_val)
```

2. **İki Koşu Karşılaştır**
```bash
# Scaling YOK
python train.py --no-scaling --exp-id exp_no_scale

# Scaling VAR
python train.py --scaling --exp-id exp_with_scale
```

3. **Analiz Yap**
```markdown
## Scaling Ablation

### Sonuçlar
| Config       | Train MSE | Val MSE | Epochs | Time  |
|--------------|-----------|---------|--------|-------|
| No Scaling   | 0.25      | 0.35    | 45     | 12s   |
| With Scaling | 0.12      | 0.18    | 25     | 8s    |

### Gözlem
- Scaling ile convergence **HIZLI** (45→25 epoch)
- MSE **DÜŞÜK** (0.35→0.18)
- Training **STABIL** (loss eğrisi smooth)

### Neden?
→ Week 0 theory: Loss yüzeyi yuvarlanır (condition number ↓)
→ GD zikzak yapmaz, doğrudan iner
```

**Kabul Kriteri:**
- ✅ Her iki koşu bitti
- ✅ Karşılaştırma tablosu var
- ✅ "Neden?" açıklaması Week 0'a referans veriyor

**Çıktı:**
```bash
git commit -m "day3: scaling ablation, 2x faster convergence"
```

---

### 📅 Gün 4: Hata Eğrisi + Over/Underfit Teşhisi

**Hedef:** Train/Val loss görselleştir, overfit örneği göster

**Görevler:**

1. **Loss Curve Plotter**
```python
import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

2. **Overfit Örneği Tetikle**
```bash
# L2 YOK, Epoch YÜKSEk
python train.py --l2 0.0 --epochs 200 --exp-id exp_overfit

# L2 VAR (normal)
python train.py --l2 1e-3 --epochs 100 --exp-id exp_normal
```

3. **Grafikleri Oluştur**
```bash
python visualize_losses.py
# Output: 
#   outputs/loss_curve_overfit.png
#   outputs/loss_curve_normal.png
```

**Beklenen Grafik (Overfit):**
```
MSE
  │
  │ ──────────╲___ Train (mükemmel)
  │            ╲
  │ ────────────╱── Val (kötüleşiyor)
  │             ↑
  │        Overfit başladı
  └────────────────→ Epoch
```

**Beklenen Grafik (Normal):**
```
MSE
  │
  │╲___
  │    ╲___ Train
  │    ╲___
  │        ╲___ Val (parallel)
  └────────────────→ Epoch
```

**Kabul Kriteri:**
- ✅ 2 grafik oluşturuldu (overfit + normal)
- ✅ Overfit örneğinde val loss yükseliyor
- ✅ Normal örnekte val loss stabil/düşüyor
- ✅ Grafikler outputs/ klasöründe

**Çıktı:**
```bash
git commit -m "day4: loss curves + overfit example"
```

---

### 📅 Gün 5: Rapor + Temizlik

**Hedef:** Profesyonel rapor yaz, test ekle

**Görevler:**

1. **Kapsamlı Rapor (week1_report.md)**

```markdown
# Week 1: Linear Regression — Final Rapor

## 🎯 Hedef
Val MSE < 0.5 başarmak ve training sürecini optimize etmek.

## 📊 Deneyler

### Özet Tablo
| Exp ID  | LR    | L2    | Scaling | Early Stop | Train MSE | Val MSE | Epochs |
|---------|-------|-------|---------|------------|-----------|---------|--------|
| exp001  | 1e-2  | 1e-3  | ✓       | ✓          | 0.15      | 0.22    | 35     |
| exp002  | 5e-3  | 1e-3  | ✓       | ✓          | 0.12      | 0.18    | 25     |
| exp003  | 1e-3  | 1e-3  | ✓       | ✓          | 0.18      | 0.25    | 50     |
| exp_ovf | 5e-3  | 0.0   | ✓       | ✗          | 0.05      | 0.45    | 200    |
| no_scl  | 5e-3  | 1e-3  | ✗       | ✓          | 0.25      | 0.35    | 45     |

### En İyi Konfigürasyon ⭐
```
LR: 5e-3
L2: 1e-3
Batch: 32
Patience: 5
Scaling: StandardScaler
```

**Sonuç:** Val MSE = 0.18 ✓

## 🧠 Neden İşe Yaradı? (Week 0 Teori Bağlantısı)

### 1. Feature Scaling
**Teori (theory_core_concepts.md, Bölüm 8):**
- Farklı ölçekler → Eliptik loss yüzeyi
- Standardizasyon → Yuvarlak yüzey
- GD zikzak yapmaz → Hızlı yakınsama

**Pratik:**
- Scaling ile 45→25 epoch (2x hızlı)
- Val MSE: 0.35→0.18 (daha iyi)

### 2. L2 Regularization
**Teori (theory_core_concepts.md, Bölüm 11):**
- L2 ← Gaussian prior (MAP)
- Büyük ağırlıkları cezalar
- Overfit'i önler

**Pratik:**
- L2=0: Val MSE patladı (0.45)
- L2=1e-3: Val MSE stabil (0.18)

### 3. Early Stopping
**Teori (theory_core_concepts.md, Bölüm 7):**
- Implicit regularization
- Val kötüleşince dur

**Pratik:**
- Patience=5 → Epoch 25'te durdu
- Overfit önlendi

### 4. Learning Rate
**Teori (theory_closure.md, Bölüm 3):**
- Çok büyük → Salınım
- Çok küçük → Yavaş
- Sweet spot: 5e-3

**Pratik:**
- 1e-2: Salınımlı (val: 0.22)
- 5e-3: Optimal (val: 0.18) ⭐
- 1e-3: Yavaş (val: 0.25)

## 📈 Grafikler

### Loss Curves
![Loss Curves](outputs/loss_curve_normal.png)

### Overfit Example
![Overfit](outputs/loss_curve_overfit.png)

**Gözlem:** L2=0 koşusunda val loss epoch 50'den sonra yükseliyor.

## 🐛 Karşılaşılan Sorunlar

### Problem 1: İlk Denemede NaN
**Sebep:** LR çok büyük (0.1)
**Çözüm:** LR'ı 0.005'e indirdim
**Teori:** theory_closure.md Bölüm 3, "LR çok büyük" semptomları

### Problem 2: Val Loss Unstable
**Sebep:** Batch size çok küçük (8)
**Çözüm:** Batch'i 32'ye çıkardım
**Teori:** Mini-batch GD dengesi

### Problem 3: Convergence Yavaş
**Sebep:** Feature scaling unutulmuş
**Çözüm:** StandardScaler eklendi
**Teori:** theory_core_concepts.md Bölüm 8

## 💭 Öğrendiklerim

1. **Theory → Practice bağlantısı GÜÇLÜ**
   - Week 0'da öğrendiğim her kavramı burada gördüm
   - MSE ← Gaussian MLE (gerçekten!)
   - L2 ← Gaussian prior (MAP)

2. **Debugging sistematik yapılmalı**
   - theory_closure.md'deki checklist işe yaradı
   - LR/Scaling/L2 sırasıyla kontrol ettim

3. **Ablation çok öğretici**
   - Scaling var/yok → Farkı NET gördüm
   - L2 var/yok → Overfit'i tetikledim

4. **Grafik görselleştirme kritik**
   - Loss curve'lere bakmadan overfit'i göremezdim
   - Sayılar tek başına yeterli değil

5. **Experiment logging disiplini**
   - exp_log.csv tutmak çok faydalı oldu
   - Hangi config'i denediğimi hatırlıyorum

## 🔄 Bir Dahaki Sefere

### Denenemedi (Zaman Yetmedi)
- [ ] Gradient clipping
- [ ] Mixed precision training
- [ ] Learning rate warmup
- [ ] Cosine annealing

### İyileştirmeler
- [ ] Otomatik hyperparameter search (Optuna?)
- [ ] Daha detaylı grafikler (weight distribution)
- [ ] Gradient norm tracking
- [ ] Confidence intervals (multiple runs)

## ➡️ Week 2 Hazırlığı

### Teknik
- [ ] MNIST dataset indir
- [ ] MLP architecture oku (theory_foundations.md'de var mı?)
- [ ] ReLU activation function teorisi

### Teori
- [ ] Non-linearity neden gerekli?
- [ ] Multi-layer backpropagation
- [ ] Classification metrics (accuracy, precision, recall)

## 📝 Notlar (Kendime)

- Linear regression'da tüm Week 0 kavramları göründü ✓
- Ablation study yaklaşımı çok işe yaradı
- Grafik çizmek sezgi geliştiriyor
- Theory olmadan debug etmek zordu (olurdu)
- Week 2'de MLP'yi aynı disiplinle yapacağım

**Val MSE: 0.18 < 0.5 ✓**
**Week 1 BAŞARILI! 🎉**
```

2. **Gradient Check Testi Ekle**
```python
# tests/test_linreg.py'a ekle

def test_gradient_numerical():
    """Autograd vs numerical gradient check."""
    model = LinearRegression(input_dim=5)
    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    
    # Autograd gradient
    loss = F.mse_loss(model(x), y)
    loss.backward()
    auto_grad = model.linear.weight.grad.clone()
    
    # Numerical gradient
    epsilon = 1e-5
    numerical_grad = torch.zeros_like(auto_grad)
    
    for i in range(auto_grad.shape[0]):
        for j in range(auto_grad.shape[1]):
            # Forward +epsilon
            model.linear.weight.data[i, j] += epsilon
            loss_plus = F.mse_loss(model(x), y)
            
            # Forward -epsilon
            model.linear.weight.data[i, j] -= 2 * epsilon
            loss_minus = F.mse_loss(model(x), y)
            
            # Restore
            model.linear.weight.data[i, j] += epsilon
            
            # Numerical gradient
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Check
    assert torch.allclose(auto_grad, numerical_grad, atol=1e-4)
```

**Kabul Kriteri:**
- ✅ week1_report.md tamamlandı
- ✅ Gradient check testi yeşil
- ✅ exp_log.csv dolu
- ✅ Tüm grafikler outputs/'ta

**Çıktı:**
```bash
git add .
git commit -m "day5: final report + gradient check, Week 1 complete!"
git tag week1-complete
```

---

## 📌 Günlük Çalışma Formatı (Tekrar Eden Ritim)

Her gün aynı disiplin:

```
1. HEDEF (1 cümle)
   "Val MSE < 0.4 yapmak"

2. PLAN (3 madde)
   - LR sweep (1e-3, 5e-3, 1e-2)
   - Early stopping on
   - L2=1e-3 sabit

3. KOŞ
   Komutları çalıştır, logla

4. KAYDET
   - exp_log.csv'ye yaz
   - Grafiği outputs/'a kaydet
   - Kısa not (5 satır)

5. COMMIT
   git commit -m "dayX: [ne yaptın], [sonuç]"
```

**Template (her gün kullan):**

```bash
# Sabah (5 dk)
echo "## Day X - $(date)" >> daily_log.md
echo "Hedef: [...]" >> daily_log.md
echo "Plan:" >> daily_log.md
echo "  1. [...]" >> daily_log.md
echo "  2. [...]" >> daily_log.md

# Akşam (5 dk)
echo "Sonuç: [...]" >> daily_log.md
echo "Öğrenim: [...]" >> daily_log.md
git commit -m "dayX: [özet]"
```

---

## 🧪 Mini Katalar (15-20 dk, Opsiyonel Ama Keskinleştirir)

### Kata 1: Gradient Check
**Hedef:** Autograd'ın doğruluğunu numeric gradient ile doğrula

```python
def gradient_check(model, x, y, epsilon=1e-5):
    """Compare autograd vs numerical gradient."""
    # Autograd
    loss = F.mse_loss(model(x), y)
    loss.backward()
    auto_grad = [p.grad.clone() for p in model.parameters()]
    
    # Numerical
    numerical_grads = []
    for param in model.parameters():
        num_grad = torch.zeros_like(param)
        for idx in np.ndindex(param.shape):
            param.data[idx] += epsilon
            loss_plus = F.mse_loss(model(x), y)
            
            param.data[idx] -= 2 * epsilon
            loss_minus = F.mse_loss(model(x), y)
            
            param.data[idx] += epsilon  # restore
            
            num_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        numerical_grads.append(num_grad)
    
    # Check
    for auto, num in zip(auto_grad, numerical_grads):
        rel_error = torch.abs(auto - num) / (torch.abs(auto) + torch.abs(num) + 1e-8)
        print(f"Max relative error: {rel_error.max().item():.2e}")
```

**Öğrenme:** Autograd'a güven (numerik doğrulama)

### Kata 2: Shape Invariants
**Hedef:** Tüm katmanlarda shape assert'leri ekle

```python
class LinearRegressionAsserted(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        assert x.dim() == 2, f"Expected 2D, got {x.dim()}D"
        assert x.shape[1] == self.input_dim, \
            f"Expected {self.input_dim} features, got {x.shape[1]}"
        
        out = self.linear(x)
        
        assert out.shape[1] == 1, f"Output should be (N, 1), got {out.shape}"
        return out
```

**Öğrenme:** Shape hataları erken yakala

### Kata 3: Seed Tekrar
**Hedef:** Aynı seed ile 3 kez koş, varyansı ölç

```python
def reproducibility_test(seed=42, n_runs=3):
    """Same config, multiple runs, measure variance."""
    results = []
    
    for run in range(n_runs):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train
        val_mse = train_model(...)
        results.append(val_mse)
    
    mean = np.mean(results)
    std = np.std(results)
    
    print(f"Val MSE: {mean:.4f} ± {std:.4f}")
    assert std < 0.01, "Too much variance! Check seeding."
```

**Öğrenme:** Reproducibility disiplini

---

## 🎯 Net Cevap: "Nasıl, Ne Zaman Pratik?"

### Bugün (45 dk)
```
✅ ŞİMDİ hızlı seti yap → BUGÜN BİTİR
   1. Setup (5 dk)
   2. Data + Manuel GD (10 dk)
   3. nn.Module + Training (15 dk)
   4. Test + Lint (10 dk)
   5. Mini rapor (5 dk)
```

### Yarın (Gün 1)
```
✅ Sprint planına göre Gün 1'i koş
   - E2E pipeline
   - Logging sistemi kur
   - İlk experiment'i kaydet
```

### Her Gün
```
✅ Günlük çalışma formatını kullan:
   1. Hedef (1 cümle)
   2. Plan (3 madde)
   3. Koş
   4. Kaydet (exp_log.csv + grafik)
   5. Commit + özet
```

### Teori Ne Zaman?
```
✅ Sadece "neden bu sonucu aldın?" kısmında
   - 3-5 maddelik açıklama
   - Week 0'a referans ver
   - Geri kalanı KOD!
```

---

## 📂 Dosya Yapısı (Week 1 Sonu)

```
week1_tensors/
├── README.md (bu dosya)
├── data_synth.py
├── linreg_manual.py
├── linreg_module.py
├── train.py
├── visualize_losses.py
├── week1_summary.md (hızlı sprint)
├── week1_report.md (detaylı rapor)
├── daily_log.md (günlük notlar)
│
├── data/
│   └── week1_linreg.pt
│
├── outputs/
│   ├── exp_log.csv
│   ├── model.pt
│   ├── metrics.json
│   ├── loss_curve_normal.png
│   └── loss_curve_overfit.png
│
└── tests/
    └── test_linreg.py
```

---

## 📋 Şablonlar

### exp_log.csv Şablonu

```csv
exp_id,lr,l2,batch,patience,scaling,train_mse,val_mse,epochs,stopped_early,time_sec,notes
exp001,5e-3,1e-3,32,5,True,0.12,0.18,25,True,8.5,baseline
exp002,1e-2,1e-3,32,5,True,0.15,0.22,35,True,10.2,lr too high
exp003,1e-3,1e-3,32,5,True,0.18,0.25,50,True,15.1,lr too low
exp_overfit,5e-3,0.0,32,999,True,0.05,0.45,200,False,45.0,L2=0 overfit
exp_no_scale,5e-3,1e-3,32,5,False,0.25,0.35,45,True,12.3,no scaling slow
```

### week1_summary.md Şablonu

```markdown
# Week 1 - Sprint Özeti

## Konfigürasyon
- LR: [...]
- L2: [...]
- Batch: [...]
- Patience: [...]
- Scaling: [Yes/No]

## Sonuçlar
- Train MSE: [...]
- Val MSE: [...] (< 0.5 ✓)
- Epochs: [...] (early stopped)
- Time: [...] sec

## Gözlem (3-5 madde)
- [...]
- [...]
- [...]

## Week 0 Bağlantısı
- [Hangi teori burada göründü?]
```

### daily_log.md Şablonu

```markdown
# Week 1 - Daily Log

## Day 1 - 2025-10-06
### Hedef
Val MSE < 0.5 yapmak

### Plan
1. Pipeline'ı koş
2. Logging sistemi kur
3. İlk experiment'i kaydet

### Sonuç
- Val MSE: 0.18 ✓
- Pipeline çalışıyor
- exp_log.csv kuruldu

### Öğrenim
- Scaling eklemek çok fark etti
- Early stopping epoch 25'te durdu

---

## Day 2 - 2025-10-07
### Hedef
[...]
```

---

## 🎓 Başarı Kriteri

### Hızlı Sprint (Bugün)
```
✅ Val MSE < 0.5
✅ Pytest yeşil
✅ week1_summary.md var
✅ Git commit yapıldı
```

### 5 Günlük Sprint (Week 1 Sonu)
```
✅ Val MSE < 0.5
✅ Tüm testler yeşil
✅ week1_report.md tamamlandı
✅ exp_log.csv dolu (5+ experiment)
✅ Loss curve grafikleri var
✅ Overfit örneği gösterildi
✅ Week 0 teori bağlantısı açıklandı
✅ Git tag: week1-complete
```

---

## 🚀 Hadi Başlayalım!

```bash
# ŞİMDİ başla (45 dk)
cd /Users/onur/code/novadev-protocol
source .venv/bin/activate
cd week1_tensors

# İlk komutu çalıştır
python data_synth.py

# Week 0'da öğrendiklerini BU KODDA GÖR! 💪
```

**Sonraki Hafta:** MLP & MNIST sınıflandırma (accuracy ≥ %90)