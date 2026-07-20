import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# 1. Carregue o dataset
# O pandas interpreta a string "None" como valor faltante (NaN); corrigimos
# para manter "None" como uma classe válida do alvo.
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv', keep_default_na=True)
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

# 2. Pré-processamento
# 2.1 Remove o identificador (não é uma feature preditiva)
df = df.drop(columns=['Person ID'])

# 2.2 "Blood Pressure" vem como texto "126/83"; dividimos em duas colunas numéricas
df[['Systolic', 'Diastolic']] = (
    df['Blood Pressure'].str.split('/', expand=True).astype(int)
)
df = df.drop(columns=['Blood Pressure'])

# 2.3 Define as features numéricas e categóricas
numeric_features = [
    'Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
    'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic',
]
categorical_features = ['Gender', 'Occupation', 'BMI Category']

X = df[numeric_features + categorical_features]
y = df['Sleep Disorder']  # alvo: None | Insomnia | Sleep Apnea (multiclasse)

# 3. Divida os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Pipeline: codifica categóricas (one-hot) e treina a árvore
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)
modelo = Pipeline([
    ('prep', preprocessor),
    ('clf', DecisionTreeClassifier(
        criterion='gini', 
        max_depth=3,              # Reduzimos de 5 para 3
        min_samples_split=10,     # Exige pelo menos 10 pacientes para tentar uma nova divisão
        min_samples_leaf=5,       # Garante que nenhuma regra final se aplique a menos de 5 pacientes
        random_state=42
    )),
])
modelo.fit(X_train, y_train)

# 5. Faça previsões
previsoes = modelo.predict(X_test)

# 6. Avalie o desempenho (multiclasse -> média ponderada)
acuracia = accuracy_score(y_test, previsoes)
precisao = precision_score(y_test, previsoes, average='weighted', zero_division=0)
revocacao = recall_score(y_test, previsoes, average='weighted', zero_division=0)
f1 = f1_score(y_test, previsoes, average='weighted', zero_division=0)

print("=== Distribuição das classes (alvo) ===")
print(y.value_counts())

print("\n=== Métricas no conjunto de teste ===")
print(f"Acurácia:  {acuracia:.2f}")
print(f"Precisão:  {precisao:.2f}")
print(f"Revocação: {revocacao:.2f}")
print(f"F1-score:  {f1:.2f}")

print("\n=== Matriz de confusão ===")
labels = sorted(y.unique())
cm = confusion_matrix(y_test, previsoes, labels=labels)
print(pd.DataFrame(cm, index=[f"real_{l}" for l in labels],
                   columns=[f"prev_{l}" for l in labels]))

print("\n=== Relatório de classificação ===")
print(classification_report(y_test, previsoes, zero_division=0))

# 7. Validação cruzada (estabilidade do modelo)
# O número de folds é limitado pela menor classe para evitar erro em classes pequenas.
n_folds = min(5, int(y.value_counts().min()))
if n_folds >= 2:
    cv_scores = cross_val_score(modelo, X, y, cv=n_folds, scoring='accuracy')
    print(f"=== Validação cruzada ({n_folds} folds) ===")
    print(f"Acurácia por fold: {cv_scores.round(2)}")
    print(f"Média: {cv_scores.mean():.2f} | Desvio padrão: {cv_scores.std():.2f}")
else:
    print("=== Validação cruzada ===")
    print("Conjunto de dados insuficiente para validação cruzada (menos de 2 amostras na menor classe).")