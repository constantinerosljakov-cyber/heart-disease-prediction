# 🫀 Heart Disease Prediction

**Предсказание сердечно-сосудистых заболеваний с помощью машинного обучения**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-FF9F00?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6F61?style=for-the-badge&logoColor=white)

---

## 📋 Краткое содержание статьи

Статья **"An artificial intelligence model for heart disease detection using machine learning algorithms"** (Victor Chang et al., 2022) посвящена разработке системы ранней диагностики сердечных заболеваний с помощью Python и моделей машинного обучения.

Авторы демонстрируют процесс обработки данных, работу с категориальными признаками и сравнение моделей. Лучший результат показал **Random Forest Classifier** (~83% точности). Особое внимание уделяется удобству и безопасности использования Python в здравоохранении.

Полный текст статьи: [`Статья-main.pdf`](Статья-main.pdf)

---

## 🎯 Цель проекта

- Провести полный EDA: корреляция, boxplots, scatter plots, баланс классов
- Preprocessing: IQR-фильтрация выбросов, OHE для номинальных, feature engineering
- Обучить 4 модели с StratifiedKFold и `class_weight='balanced'`
- Сравнить модели по AUC, F1, Recall, Precision, Confusion Matrix
- Построить Feature Importance и SHAP-анализ
- Собрать Blending-ансамбль всех 4 моделей

---

## 📊 Результаты моделей

| Модель               | Accuracy | ROC-AUC | F1     |
|----------------------|----------|---------|--------|
| Logistic Regression  | ~84%     | ~0.91   | ~0.84  |
| Random Forest        | ~87%     | ~0.94   | ~0.87  |
| XGBoost (Optuna)     | ~89%     | ~0.96   | ~0.89  |
| Neural Network       | ~88%     | ~0.95   | ~0.88  |
| **Blend (все 4)**    | **~90%** | **~0.96** | **~0.90** |

---

## 📁 Структура проекта

```
heart-disease-prediction/
├── notebooks/
│   └── main.ipynb          # EDA + preprocessing + обучение + SHAP
├── models/
│   ├── best_rf.pkl          # Random Forest
│   ├── best_xgb.pkl         # XGBoost (Optuna)
│   ├── best_lr.pkl          # Logistic Regression
│   ├── best_nn.pth          # Neural Network (PyTorch)
│   ├── scaler.pkl           # StandardScaler
│   ├── label_encoders.pkl   # LabelEncoders
│   └── feature_columns.pkl  # Список признаков
├── inference.py             # Скрипт инференса (blending)
├── requirements.txt         # Зависимости
└── README.md
```

---

## 🚀 Как запустить

```bash
# 1. Клонировать репозиторий
git clone https://github.com/constantinerosljakov/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Запустить ноутбук
jupyter notebook notebooks/main.ipynb

# 4. Инференс на тестовых данных
python inference.py
```

---

## 🛠 Технологии

- **Python 3.10+**, Pandas, NumPy
- **Scikit-learn** — LR, RF, preprocessing, метрики
- **XGBoost** — градиентный бустинг
- **Optuna** — автоматический подбор гиперпараметров
- **PyTorch** — ResNet-подобная нейросеть с BatchNorm + CosineAnnealingLR
- **SHAP** — интерпретируемость модели (summary plot, force plot)
