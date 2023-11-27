import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# importação do arquivo
df = pd.read_csv('mail_data.csv')

# listando apenas os dados que não são nulos
data = df.where((pd.notnull(df)), '')

# definindo os valores para spam e ham (email comum)
data.loc[data['Category'] == 'spam', 'Category',] = 1
data.loc[data['Category'] == 'ham', 'Category',] = 0

# x será a coluna de mensagem e y, a de categoria
x = data['Message']
y = data['Category']

# fazendo o treinamento
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# preparando os dados para serem usados em um modelo de aprendizado de máquina
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# regressão logística é frequentemente utilizada para problemas de classificação
model = LogisticRegression()

# treinando o modelo de regressão logística usando o conjunto de treinamento
model.fit(x_train_features, y_train)

# função para classificar o texto inserido
def classificar_texto():
    texto = entrada_texto.get()

    if not texto:
        messagebox.showwarning("Aviso", "Por favor, insira um texto para classificação.")
        return

    # vetorizar o texto
    texto_vetorizado = feature_extraction.transform([texto])

    # prever a probabilidade
    probabilidade_spam = model.predict_proba(texto_vetorizado)[0, 1]

    # ajustar o limiar de decisão
    limiar = 0.5  # ajuste conforme necessário
    resultado = 1 if probabilidade_spam > limiar else 0

    # exibir resultado na GUI
    if resultado == 0:
        resultado_label.config(text="Resultado: E-mail comum", font=("underline",))
    else:
        resultado_label.config(text="Resultado: SPAM!", font=("underline",))

# criar a janela principal
janela = tk.Tk()
janela.title("Detector de Spam")
janela.configure(bg="white")  # Cor de fundo da janela

# criando as labels, input e botão
frame_titulo = tk.Frame(janela, bg="blue")
rotulo = tk.Label(frame_titulo, text="Insira o texto suspeito recebido:", bg="blue", fg="white")
entrada_texto = tk.Entry(janela, width=50)
botao_classificar = tk.Button(janela, text="Classificar", command=classificar_texto, bg="#4CAF50", fg="white", padx=20, pady=10, relief=tk.GROOVE, font=("Helvetica", 12, "bold"))
resultado_label = tk.Label(janela, text="Resultado:")

# organizando os itens na janela
frame_titulo.pack(fill="x")
rotulo.pack(pady=10)
entrada_texto.pack(pady=10)
botao_classificar.pack(pady=10)
resultado_label.pack(pady=10)

# mostrando a janela
janela.mainloop()
