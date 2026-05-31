"""
inference.py — Heart Disease Prediction
Использует лучшую модель (XGBoost) + blending всех 4 моделей.
"""
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ====================== АРХИТЕКТУРА NN (должна совпадать с обучением) ======================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.block(x))

class HeartNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.res1  = ResidualBlock(256)
        self.down  = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.res2  = ResidualBlock(128)
        self.out   = nn.Linear(128, 2)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        return self.out(x)

# ====================== ЗАГРУЗКА ДАННЫХ ======================
print("Загружаем тестовые данные...")
test     = pd.read_csv('../data/test.csv')
test_ids = test['ID'].copy()   # сохраняем до любых преобразований

# ====================== ПРЕДОБРАБОТКА (идентична ноутбуку) ======================

# 1. Округление chest
test['chest'] = test['chest'].round().astype(int)

# 2. Feature Engineering
def add_features(df):
    df = df.copy()
    df['age_x_bp']       = df['age'] * df['resting_blood_pressure']
    df['hr_reserve']     = (220 - df['age']) - df['maximum_heart_rate_achieved']
    df['oldpeak_per_hr'] = df['oldpeak'] / (df['maximum_heart_rate_achieved'] + 1e-5)
    df['chol_per_age']   = df['serum_cholestoral'] / (df['age'] + 1e-5)
    df['is_elderly']     = (df['age'] >= 60).astype(int)
    df['sex_x_age']      = df['sex'].astype(str) + '_' + pd.cut(
        df['age'], bins=[0,45,55,65,100], labels=['<45','45-55','55-65','65+']
    ).astype(str)
    return df

test = add_features(test)

# 3. One-Hot Encoding номинальных
nominal_cols = ['chest', 'thal', 'sex_x_age']
test = pd.get_dummies(test, columns=nominal_cols, drop_first=False)

# 4. Label Encoding
label_encoders = joblib.load('../models/label_encoders.pkl')
ordinal_cols = ['sex', 'fasting_blood_sugar', 'resting_electrocardiographic_results',
                'exercise_induced_angina', 'slope', 'number_of_major_vessels']
for col in ordinal_cols:
    test[col] = label_encoders[col].transform(test[col])

# 5. Выравниваем колонки под обученные модели
feature_columns = joblib.load('../models/feature_columns.pkl')
test_feat = test.reindex(columns=['ID'] + feature_columns, fill_value=0)

# 6. Убираем ID
X_test = test_feat[feature_columns]

# 7. Масштабирование
scaler   = joblib.load('../models/scaler.pkl')
num_cols = ['age', 'resting_blood_pressure', 'serum_cholestoral',
            'maximum_heart_rate_achieved', 'oldpeak',
            'age_x_bp', 'hr_reserve', 'oldpeak_per_hr', 'chol_per_age']
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ====================== ЗАГРУЗКА МОДЕЛЕЙ ======================
print("Загружаем модели...")
rf_model  = joblib.load('../models/best_rf.pkl')
xgb_model = joblib.load('../models/best_xgb.pkl')
lr_model  = joblib.load('../models/best_lr.pkl')

nn_model  = HeartNN(input_dim=len(feature_columns))
nn_model.load_state_dict(torch.load('../models/best_nn.pth', map_location='cpu'))
nn_model.eval()

# ====================== ПРЕДСКАЗАНИЕ ======================
print("Предсказываем...")
prob_lr  = lr_model.predict_proba(X_test)[:,1]
prob_rf  = rf_model.predict_proba(X_test)[:,1]
prob_xgb = xgb_model.predict_proba(X_test)[:,1]

with torch.no_grad():
    nn_out  = nn_model(torch.tensor(X_test.values, dtype=torch.float32))
    prob_nn = F.softmax(nn_out, dim=1)[:,1].numpy()

# Blending: усредняем вероятности
prob_blend  = (prob_lr + prob_rf + prob_xgb + prob_nn) / 4
pred_blend  = (prob_blend >= 0.5).astype(int)

# ====================== SUBMISSION ======================
submission = pd.DataFrame({
    'ID':    test_ids,
    'class': pred_blend
})
submission.to_csv('../submission.csv', index=False)

print(f"\nГотово! submission.csv сохранён.")
print(f"Предсказаний: {len(submission)}")
print(f"Распределение: {submission['class'].value_counts().to_dict()}")
print(submission.head(10).to_string(index=False))
