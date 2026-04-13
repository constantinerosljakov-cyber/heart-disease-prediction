import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ====================== НЕЙРОННАЯ СЕТЬ ======================
class NN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.model(x)

# ====================== ЗАГРУЗКА МОДЕЛИ ======================
print("Загружаем лучшую модель...")

# Можно использовать Random Forest или Neural Network
# Сейчас используем Random Forest (быстрее и стабильно)
model = joblib.load('../models/best_rf.pkl')

# Если хочешь использовать Neural Network — раскомментируй ниже:
# model = NN(input_dim=...) 
# model.load_state_dict(torch.load('../models/best_nn.pth'))
# model.eval()

# ====================== ЗАГРУЗКА ТЕСТОВЫХ ДАННЫХ ======================
test = pd.read_csv('../data/test.csv')

# Предобработка (точно такая же, как в ноутбуке)
cat_cols = ['sex', 'chest', 'fasting_blood_sugar', 'resting_electrocardiographic_results',
            'exercise_induced_angina', 'slope', 'number_of_major_vessels', 'thal']

test = pd.get_dummies(test, columns=cat_cols, drop_first=True)

# Загружаем scaler (если используешь Neural Network или Logistic Regression)
# scaler = joblib.load('../models/scaler.pkl')   # если сохранял scaler

X_test = test.drop(['ID'], axis=1)

print("Предсказываем...")

# Предсказание
predictions = model.predict(X_test)

# Создаём submission
submission = pd.DataFrame({
    'ID': test['ID'],
    'class': predictions
})

submission.to_csv('../submission.csv', index=False)
print("Готово! Файл submission.csv сохранён в главной папке проекта.")