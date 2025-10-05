# Hafta 1: Tensör, Autograd, Linear Regression

**Hedef:** PyTorch'un temellerini anla, sıfırdan linear regression yaz, train/val split + regularization ekle.

---

## 📚 Teori (45 dakika)

### Öğrenilecekler:
1. **Tensor nedir?** Çok boyutlu array (numpy benzeri ama GPU desteği).
2. **Autograd:** Otomatik türev hesaplama (backpropagation için).
3. **Optimizer:** SGD, Adam (gradient descent varyasyonları).
4. **Loss fonksiyonu:** MSE (Mean Squared Error).
5. **Regularization:** L2 (weight decay) → overfitting'i önle.

### Kaynaklar:
- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Linear Regression Explained](https://www.youtube.com/watch?v=nk2CQITm_eo) (3Blue1Brown style)

---

## 💻 Pratik (90 dakika)

### Adım 1: Manuel Linear Regression (ilkel API)

**Görev:** `y = wx + b` formülünü PyTorch tensörleri ile yaz.

Dosya: `linreg_manual.py`

```python
# Pseudo-code:
# 1. Rastgele data üret: X, y
# 2. Parametreleri başlat: w, b (requires_grad=True)
# 3. Loop:
#    a. Forward: y_pred = X @ w + b
#    b. Loss: mse = ((y_pred - y) ** 2).mean()
#    c. Backward: loss.backward()
#    d. Update: w -= lr * w.grad (manual SGD)
#    e. Zero grad: w.grad.zero_()
```

### Adım 2: PyTorch nn.Module ile (daha temiz)

Dosya: `linreg_module.py`

```python
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

### Adım 3: Train/Val Split + L2 Regularization

Dosya: `train.py`

- %80 train, %20 val split
- Her epoch sonunda val loss logla
- Early stopping (5 epoch boyunca iyileşme yoksa dur)
- L2 reg: `optimizer = Adam(..., weight_decay=0.01)`

---

## 📊 Teslim (Hafta 1 Sonu)

### 1. Jupyter Notebook: `linreg_from_scratch.ipynb`

İçerik:
- Data oluşturma (sentetik linear data)
- Manuel regresyon (adım adım)
- nn.Module regresyon
- Train/val curves (matplotlib)
- Final sonuç: `MSE < 0.5` (sentetik data için)

### 2. Test Dosyası: `tests/test_linreg.py`

```python
def test_model_convergence():
    """Model train edildiğinde MSE düşmeli."""
    # ... train loop
    assert final_mse < 0.5

def test_l2_regularization():
    """L2 reg ile weight'ler küçülmeli."""
    # ... compare with/without weight_decay
    assert norm_with_l2 < norm_without_l2
```

Çalıştır:
```bash
pytest tests/test_linreg.py -v
```

### 3. Kapanış Raporu: `week1_summary.md`

```markdown
# Hafta 1: Tensör & Linear Regression Özeti

## 🎯 Tamamlananlar
- [ ] Manuel gradient descent
- [ ] nn.Module ile model
- [ ] Train/val split
- [ ] L2 regularization
- [ ] Testler geçti (MSE < 0.5)

## 📈 Sonuçlar
- Final Train MSE: X.XX
- Final Val MSE: X.XX
- Epoch sayısı: XX

## 💭 Öğrendiklerim
- [3 madde: en önemli insight'lar]

## 🐛 Karşılaşılan Sorunlar
- [Varsa sorunlar ve çözümler]

## ➡️ Hafta 2 Hazırlığı
- MNIST dataset indir
- MLP (multi-layer perceptron) teorisi oku
```

---

## 🎯 Başarı Kriteri

✅ **MSE < 0.5** (sentetik linear dataset için)
✅ **Testler geçmeli:** `pytest tests/test_linreg.py`
✅ **Loss curve grafiği:** Train ve val loss beraber plot edilmeli

---

## 🔧 Starter Kod

`linreg_manual.py` içinde iskelet hazır. Eksik kısımları sen dolduracaksın (🚧 TODO işaretleri var).

---

**Sonraki Hafta:** MLP & MNIST sınıflandırma (accuracy ≥ %90)
