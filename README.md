# Heart Disease Prediction (PJ)

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