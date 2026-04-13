# 🫀 Heart Disease Prediction

**Предсказание сердечно-сосудистых заболеваний с помощью машинного обучения**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-FF9F00?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

### 📋 Краткое содержание научной статьи

Статья **"An artificial intelligence model for heart disease detection using machine learning algorithms"** (Victor Chang et al., 2022) посвящена разработке системы ранней диагностики сердечных заболеваний с помощью Python и моделей машинного обучения.

Авторы демонстрируют процесс обработки данных, работу с категориальными признаками и сравнение моделей. Лучший результат показал **Random Forest Classifier** (~83% точности). Особое внимание уделяется удобству и безопасности использования Python в здравоохранении.

Полный текст статьи: [`Статья-main.pdf`](Статья-main.pdf)

---

### 🎯 Цель проекта

- Изучить статью и применить полученные знания
- Провести анализ данных (корреляция + выбросы)
- Обучить 3 модели:
  - Logistic Regression
  - Random Forest
  - Neural Network (PyTorch)
- Сравнить результаты и подготовить модель для инференса

---

### 📊 Результаты моделей

| Модель                  | Accuracy | Примечание                  |
|-------------------------|----------|-----------------------------|
| Logistic Regression     | ~84.2%   | Базовая линейная модель     |
| Random Forest           | ~87.1%   | Лучшая классическая модель  |
| **Neural Network**      | **~88.4%** | **Лучшая модель**           |

**Лучшая модель:** Полносвязная нейронная сеть на PyTorch

---

### 📁 Структура проекта
heart-disease-prediction/
├── notebooks/
│   └── main.ipynb                 # Основной ноутбук с анализом и обучением
├── models/
│   └── best_model.pth             # Лучшие веса нейронной сети
├── inference.py                   # Скрипт для предсказания
├── requirements.txt               # Необходимые библиотеки
├── README.md                      # Этот файл
└── data/                          # (не загружается в GitHub)
text---

### 🚀 Как запустить проект

```bash
# 1. Клонировать репозиторий
git clone https://github.com/ВАШ_НИК/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Запустить ноутбук
jupyter notebook notebooks/main.ipynb# Heart Disease Prediction (PJ)

**Предсказание сердечно-сосудистых заболеваний с помощью машинного обучения**

## Краткое содержание научной статьи

Статья **"An artificial intelligence model for heart disease detection using machine learning algorithms"** (Victor Chang и др., 2022) описывает разработку системы раннего выявления сердечных заболеваний с помощью Python и моделей машинного обучения.

Авторы демонстрируют обработку данных, преобразование категориальных признаков, обучение моделей (Logistic Regression и Random Forest). Лучший результат показал **Random Forest** (~83% точности). Подчёркивается удобство Python для медицинских приложений.

Полный текст статьи: `Статья-main.pdf`

## Результаты моделей

- Logistic Regression: ~84%
- Random Forest: ~87%
- Neural Network (PyTorch): ~88%

**Лучшая модель:** Random Forest / Neural Network

## Как запустить проект

```bash
pip install -r requirements.txt
python inference.py
