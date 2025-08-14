---
title: Cbr Rate Demo
emoji: 📈
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
python_version: 3.12.0
app_file: app.py
pinned: false
license: mit
---

Репозиторий на [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/ruslantur77/cbr-rate-demo/)

# CBR Key-Rate Forecast  
**Bidirectional LSTM** для прогноза **ключевой ставки ЦБ РФ**.

### 🚀 Как пользоваться
1. Введите **4 последних месяца** значений:
   - курс USD (на конец месяца)  
   - годовая инфляция (%)  
   - ключевая ставка (%)  
2. Нажмите **Predict**  
3. Получите прогноз **ключевой ставки** на **следующее заседание** в % годовых.

### ⚙️ Модель
- **Framework**: TensorFlow 2.x  
- **Architecture**: Bidirectional LSTM (32 units)  
- **Input**: 11 признаков (4 USD + 4 CPI + 3 KS)  
- **Val MSE**: ≈ 0.6636 на истории 2013-2025
- **MAE** ≈ 1.22
- **RMSE** ≈ 1.98
- **MAPE** ≈ 10.22 %
- **DirAcc** ≈ 34.6 % 
- **Scaler**: RobustScaler (fit on train only)

### 📦 Файлы
- `rate_bidir_lstm.h5` – обученная модель  
- `scaler_X_lstm.gz` / `scaler_y_lstm.gz` – скейлеры  
- `app.py` – Gradio-интерфейс

### 📝 Примеры
- `USD: [97.8, 88.3, 85.5, 81.5]`  
- `CPI: [10.06, 10.34, 10.23, 9.88]`  
- `KS: [21.0, 20.0, 18.0]`  
→ **Прогноз: 15.66%**

### 🐛 Issues & PR
Приветствуются! Открывайте Issues или Pull Request в этом репозитории.

### ⚠️ Дисклеймер
Все вышесказанное не является индивидуальной инвестиционной рекомендацией.