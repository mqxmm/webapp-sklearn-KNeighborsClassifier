import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# Загрузка данных из файла
df = pd.read_excel('data7.xlsx')

# Разделение данных на признаки и целевую переменную
X = df.drop(columns=['format'])
y = df['format'].values

# Создание модели классификации
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X, y)

# Ввод новых данных aka веб интерфейс
st.sidebar.header('Entering parameters')
def user_input_features():
    complexity = st.sidebar.slider('complexity', 0.01, 1.00, 0.43)
    level = st.sidebar.slider('level', 1, 3, 2)
    data = {'complexity': complexity,
            'level': level}
    features = pd.DataFrame(data, index=[0]).to_numpy()
    print(features)
    return features
new_data = user_input_features()
st.write(new_data)
#new_data = np.arange(10).reshape((5, 2)), range(5)
# Предсказание формата тренировки для новых данных

prediction = knn.predict(new_data)
st.sidebar.subheader(f'Format of training: {prediction[0]}')
