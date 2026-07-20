# Tree Decision

Treinamento de uma árvore de decisão com `scikit-learn`, gerenciado pelo `uv`.

## O que o modelo classifica

O modelo é um **classificador** cujo objetivo é prever a **variável alvo `Sleep Disorder`** (distúrbio do sono) de uma pessoa a partir de seus dados de saúde e estilo de vida. Ou seja, dado um conjunto de características de uma pessoa, o modelo responde: *"essa pessoa apresenta qual distúrbio do sono?"*

A variável alvo possui **3 classes**:

| Classe | Significado |
| --- | --- |
| `None` | Nenhum distúrbio do sono detectado |
| `Insomnia` | Insônia (dificuldade para adormecer ou manter o sono) |
| `Sleep Apnea` | Apneia do sono (interrupções da respiração durante o sono) |

> No arquivo `Sleep_health_and_lifestyle_dataset.csv` os casos sem distúrbio vêm como valores ausentes e são preenchidos com `None` no pré-processamento.

## Dataset

O arquivo `Sleep_health_and_lifestyle_dataset.csv` reúne dados de saúde e hábitos de sono. As colunas utilizadas como **entradas (features)** do modelo são:

- **Numéricas:** `Age` (27–59), `Sleep Duration` (5.8–8.5 h), `Quality of Sleep` (4–9), `Physical Activity Level` (30–90), `Stress Level` (3–8), `Heart Rate` (65–86 bpm), `Daily Steps` (3000–10000), `Systolic` e `Diastolic` (extraídas de `Blood Pressure`, no formato "sistólica/diastólica").
- **Categóricas:** `Gender` (`Female`, `Male`), `Occupation` (Accountant, Doctor, Engineer, Lawyer, Manager, Nurse, Sales Representative, Salesperson, Scientist, Software Engineer, Teacher), `BMI Category` (`Normal`, `Normal Weight`, `Overweight`, `Obese`).

A coluna `Person ID` é descartada, pois não agrega informação preditiva.

## Requisitos

- [uv](https://docs.astral.sh/uv/) instalado
- Python >= 3.10

## Como usar

```bash
# Criar o ambiente e instalar as dependências
uv sync

# Rodar o treinamento
uv run train.py
```

## Estrutura

- `train.py` — script de treinamento da árvore de decisão
- `pyproject.toml` — configuração do projeto e dependências
