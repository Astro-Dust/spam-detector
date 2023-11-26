# -*- coding: utf-8 -*-
"""checagem_spam.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f-eYmj1Zaq2KYhKejkkrQYAm1gBNaKTB
"""

# importação das bibliotecas que iremos utilizar
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# montagem do google drive
from google.colab import drive
drive.mount('/content/drive')

# Obs: ham -> não é spam

# importação do arquivo
df = pd.read_csv('caminho/do/arquivo/no/drive/mail_data.csv')
#display(df)

# listando apenas os dados que não são nulos
data = df.where((pd.notnull(df)), '')

# mostrando informações do dataframe
#data.info()

# mostrando as dimensões
#data.shape

# definindo os valores para spam e ham (email comum)
data.loc[data['Category'] == 'spam', 'Category',] = 1
data.loc[data['Category'] == 'ham', 'Category',] = 0

# x será a coluna de mensagem e y, a de categoria
x = data['Message']
y = data['Category']

# fazendo o treinamento
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=3)

# preparando os dados para serem usados em um modelo de aprendizado de máquina
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english',lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# regressão logística é frequentemente utilizada para problemas de classificação
model = LogisticRegression()

# treinando o modelo de regressão logística usando o conjunto de treinamento, onde o modelo aprende a relação entre as características e os rótulos fornecidos
model.fit(x_train_features, y_train)

# calculando as previsões do modelo no conjunto de treinamento e, em seguida, avaliando a acurácia dessas previsões
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)

print('Accuracy on training data: ', accuracy_on_training_data)
prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print('Accuracy on test data: ', accuracy_on_test_data)

# testando o programa e mostrando a classificação do texto do e-mail
input_your_mail = ['free vacancy at venice']
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)

if (prediction[0] == 0):
  print("E-mail comum.")
else:
  print("SPAM")

# 0 -> não é spam
# 1 -> é spam