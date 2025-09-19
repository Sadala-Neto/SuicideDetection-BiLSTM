#region Bibliotecas utilizadas
import pandas as pd
import numpy as np
import re
import nltk


from sklearn.metrics import confusion_matrix, classification_report, f1_score,roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, Dropout, Input, BatchNormalization, MaxPooling1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from keras.optimizers import Adam

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# endregion


# region Seleção do dataset

print("====================================")
print("Inicio da fase de seleção do dataset")
print("====================================")

# Definindo os caminhos para os arquivos do dataset de suicídio
df_suicide_data = pd.read_csv("../Suicide_Detection.csv")

# Mapeando as colunas do dataset de suicídio para 'text' e 'class'
df_suicide_data = df_suicide_data[['text', 'class']]

print("Primeiras 5 linhas do DataFrame original")
print(df_suicide_data.head())

# Trocando os valores da classe para o padrão numérico (0 para não-suicídio, 1 para suicídio)
# classe: 'non-suicide' -> 0, 'suicide' -> 1
label_map = {'non-suicide': 0, 'suicide': 1}

df_suicide_data['feeling'] = df_suicide_data['class'].replace(label_map)

# Descartando a coluna 'class' original e renomeando a nova coluna de mapeamento
df_suicide_data = df_suicide_data.drop(columns='class')

print("\nDataFrame após mapeamento de classes")
print(df_suicide_data.head())

print(f"\nShape do DataFrame combinado (único arquivo): {df_suicide_data.shape}")
print("\nContagem de classes no DataFrame combinado (antes da remoção de duplicatas)")
print(df_suicide_data['feeling'].value_counts())

# Remover duplicatas da coluna 'text'
print("\nRemovendo duplicatas da coluna 'text'")
initial_rows = len(df_suicide_data)
df_suicide_data.drop_duplicates(subset=['text'], inplace=True)
print(f"Número de linhas antes: {initial_rows}")
print(f"Número de linhas após remover duplicatas: {len(df_suicide_data)}")
print("\nContagem de classes após remover duplicatas:")
print(df_suicide_data['feeling'].value_counts())

# Balanceamento após a remoção de duplicatas
print("\nContagem de classes após remover duplicatas (antes do balanceamento):")
initial_counts = df_suicide_data['feeling'].value_counts()
print(initial_counts)

# Separa as classes
df_class_0 = df_suicide_data[df_suicide_data['feeling'] == 0]  # non-suicide
df_class_1 = df_suicide_data[df_suicide_data['feeling'] == 1]  # suicide

# Encontra o tamanho da menor classe para balanceamento
min_class_size = min(len(df_class_0), len(df_class_1))

# Reduz o número de amostras das classes para o tamanho da menor
df_class_0_balanced = df_class_0.sample(n=min_class_size, random_state=42)
df_class_1_balanced = df_class_1.sample(n=min_class_size, random_state=42)

# Concatena as classes balanceadas e embaralha o dataset final
df_balanced = pd.concat([df_class_0_balanced, df_class_1_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

# Atualiza df_suicide_data com o dataset balanceado
df_suicide_data = df_balanced

print("\nContagem de classes após o balanceamento:")
print(df_suicide_data['feeling'].value_counts())
print(f"Shape do DataFrame final após balanceamento: {df_suicide_data.shape}")

print("====================================")
print("Fim da fase de seleção do dataset")
print("====================================")

# endregion


# region Pre-processamento

from bs4 import BeautifulSoup

print("\n====================================")
print("Inicio do Pré-processamento")
print("====================================")

# Pré-processamento dos dados
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remover tags HTML (o dataset Amazon não tem, mas manter é seguro)
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    # Remover links
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text), flags=re.MULTILINE)
    # Remover menções a usuários (@)
    text = re.sub(r'\@\w+|\#', '', str(text))
    # Converter para minúsculas
    text = text.lower()
    # Remover pontuações e caracteres especiais
    text = re.sub(r'[^\w\s]', '', str(text))
    # Remover múltiplos espaços em branco e espaços nas bordas
    text = ' '.join(text.split())
    # Remover stop words
    # ⚠️ IMPORTANTE:
    # - Mantenha esta linha ativa → versão SEM stopwords
    # - Comente esta linha → versão COM stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df_suicide_data['text'] = df_suicide_data['text'].apply(preprocess_text)
print("\nDataFrame após pré-processamento de texto:")
print(df_suicide_data.head())

# Total de amostras após duplicatas no df_suicide_data
total_samples = len(df_suicide_data)

# Teste (20% do total)
test_size_final = 0.2
# Validação (25% do restante, que é 80% do total, então 20% do total)
val_size_final = 0.25 / (1 - test_size_final) * test_size_final # 0.25 * 0.8 = 0.2 do total
val_size_from_train = 0.25

# Dados para divisão
X = df_suicide_data['text'].values
y = df_suicide_data['feeling'].values
le = LabelEncoder()
y = le.fit_transform(y)

# Primeira divisão: separa Teste (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size_final, random_state=5, stratify=y)
# Segunda divisão: do restante (X_temp, y_temp), separa Validação (25% dele, que é 20% do total) e Treino (75% dele)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_from_train, random_state=37, stratify=y_temp)

print(f"\nNúmero de amostras no conjunto de Treino: {len(X_train)}")
print(f"Número de amostras no conjunto de Validação: {len(X_val)}")
print(f"Número de amostras no conjunto de Teste: {len(X_test)}")

print("\nDistribuição de sentimentos no conjunto de Treino:")
print(pd.Series(y_train).value_counts(normalize=True))
print("Distribuição de sentimentos no conjunto de Validação:")
print(pd.Series(y_val).value_counts(normalize=True))
print("Distribuição de sentimentos no conjunto de Teste:")
print(pd.Series(y_test).value_counts(normalize=True))


# Vetorização dos textos
tokenizer = Tokenizer(num_words=50000, oov_token="<unk>")
tokenizer.fit_on_texts(X_train)

# Convertendo os textos pré-processados em sequências numéricas de tokens
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_val = tokenizer.texts_to_sequences(X_val)

print("\n=== Análise do Comprimento das Sequências ===")

# Calcular os comprimentos das sequências em cada conjunto
train_lengths = [len(seq) for seq in X_train]
val_lengths = [len(seq) for seq in X_val]
test_lengths = [len(seq) for seq in X_test]

vocab_size = len(tokenizer.word_index) + 1
max_len = 160 # sem stopwords = 160; com stopwords = 320

# Aplica o padding
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)
X_val = pad_sequences(X_val, padding='post', maxlen=max_len)

print(f"\nShape de X_train após padding: {X_train.shape}")
print(f"Shape de X_test após padding: {X_test.shape}")
print(f"Shape de X_val após padding: {X_val.shape}")

print("\n====================================")
print("Fim do Pré-processamento")
print("====================================")

# endregion


# region Analise usando a RNA

#criação do modelo
model = Sequential()
model.add(Input(shape=(max_len,)))
model.add(Embedding(vocab_size, 100)) 
model.add(Dropout(0.3))
model.add(Conv1D(32, 5, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001))))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'precision', 'recall', 'auc'])

model.summary()

# Calcular pesos de classe automaticamente
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

print("Pesos de classe:", class_weights)
print()

#criterio de parada & Define o callback EarlyStopping
early_stopping_loss = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
callbacks_list = [early_stopping_loss, reduce_lr]

#Treinamento do modelo
print("Inicio do treinamento")
rna = model.fit(X_train, y_train, epochs=25, batch_size=1000, validation_data=(X_val, y_val), callbacks=callbacks_list, class_weight=class_weights) 

def avaliar_modelo(model, X, y, nome_conjunto="Teste"):
    print(f"\n Avaliação no conjunto de {nome_conjunto.upper()}:")
    resultados = model.evaluate(X, y, verbose=0)
    print(f"Loss:       {resultados[0]:.4f}")
    print(f"Acurácia:   {resultados[1]:.4f}")
    print(f"Precisão:   {resultados[2]:.4f}")
    print(f"Revocação:  {resultados[3]:.4f}")
    print(f"AUC:        {resultados[4]:.4f}")

    # Predição
    y_probs = model.predict(X).flatten()
    y_pred = np.round(y_probs)

    # Matriz de confusão
    print(f"\n🔹 Matriz de Confusão - {nome_conjunto}")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred, digits=4))

    # F1-score explícito
    f1 = f1_score(y, y_pred)
    print(f"F1-Score ({nome_conjunto}): {f1:.4f}")

# Chamada para treino, validação e teste
avaliar_modelo(model, X_train, y_train, "Treinamento")
avaliar_modelo(model, X_val, y_val, "Validação")
avaliar_modelo(model, X_test, y_test, "Teste")

print("====================================")
print("Fim da Analise usando a RNA")
print("====================================")


# endregion


