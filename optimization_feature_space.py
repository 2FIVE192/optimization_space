import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Загрузка данных
data = pd.read_csv('titanic.csv')

# 1. Оценка пропусков, дубликатов, категориальных переменных
print("Пропуски в данных:\n", data.isnull().sum())
print("\nКоличество дубликатов:", data.duplicated().sum())

# Определение категориальных переменных
categorical_features = data.select_dtypes(include=['object']).columns
print("\nКатегориальные переменные:", categorical_features)

# Значения категориальных переменных
for feature in categorical_features:
    print(f"\nЗначения для {feature}:\n", data[feature].value_counts())

# Корреляционная матрица для числовых столбцов
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
print("\nКорреляционная матрица:\n", correlation_matrix)

# Визуализация корреляционной матрицы для лучшего анализа
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Корреляционная матрица числовых признаков")
plt.show()

# 2. Заполнение пропусков
# a) Медианное заполнение для Age и Fare
imputer_median = SimpleImputer(strategy='median')
data['Age'] = imputer_median.fit_transform(data[['Age']])
data['Fare'] = imputer_median.fit_transform(data[['Fare']])

# b) Заполнение KNN для числовых значений
imputer_knn = KNNImputer(n_neighbors=5)
data[['Age', 'Fare']] = imputer_knn.fit_transform(data[['Age', 'Fare']])

# c) Заполнение модой для Cabin и Embarked (используем .ravel() для одномерного результата)
imputer_mode = SimpleImputer(strategy='most_frequent')
data['Cabin'] = imputer_mode.fit_transform(data[['Cabin']]).ravel()
data['Embarked'] = imputer_mode.fit_transform(data[['Embarked']]).ravel()

# 3. Кодирование категориальных переменных
# a) One-hot encoding для Embarked
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# b) Label encoding для Sex
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

# c) Target encoding для Cabin с использованием средней стоимости Fare
data['Cabin_encoded'] = data.groupby('Cabin')['Fare'].transform('mean')
data.drop('Cabin', axis=1, inplace=True)

# 4. Оценка и удаление выбросов
# Выявление выбросов с помощью Z-оценки

# Масштабируем числовые данные для визуализации выбросов
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))
scaled_data_df = pd.DataFrame(scaled_data, columns=data.select_dtypes(include=[np.number]).columns)

# Визуализация выбросов перед удалением (ящик с усами для масштабированных данных)
plt.figure(figsize=(12, 6))
sns.boxplot(data=scaled_data_df, showfliers=False)  # Отключаем отображение точек выбросов
plt.xticks(rotation=90, ticks=range(len(scaled_data_df.columns)), labels=scaled_data_df.columns, fontsize=10)  # Задаем метки для каждого признака с меньшим шрифтом
plt.title("Распределение признаков перед удалением выбросов (масштабированные данные)")
plt.show()


# Удаление выбросов
z_scores = np.abs(zscore(data.select_dtypes(include=[np.number])))
data_clean = data[(z_scores < 3).all(axis=1)].copy()

# 5. Сжатие признаков методом PCA до 3 компонент
pca = PCA(n_components=3)
principal_components = pca.fit_transform(data_clean.select_dtypes(include=[np.number]))

# Создаем DataFrame с компонентами PCA и объединяем его с data_clean
pca_df = pd.DataFrame(principal_components, columns=['PCA1', 'PCA2', 'PCA3'], index=data_clean.index)
data_clean = pd.concat([data_clean, pca_df], axis=1)

# 6. Визуализация данных после PCA
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(data_clean['PCA1'], data_clean['PCA2'], data_clean['PCA3'], c=data_clean['Sex'], cmap='coolwarm')
plt.colorbar(sc, label='Sex')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.title("Визуализация данных после PCA")
plt.show()
