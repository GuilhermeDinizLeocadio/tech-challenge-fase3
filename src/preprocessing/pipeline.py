# =============================================================
# PIPELINE — Pré-processamento integrado ao modelo
#
# Define o ColumnTransformer que trata:
#   - Numéricas: imputação (mediana) + padronização (scaling)
#   - Categórica (uf): one-hot encoding
#
# O pré-processamento é encapsulado num Pipeline do Scikit-learn,
# de modo que a imputação/scaling são aprendidos SÓ no treino
# e aplicados no teste — evitando data leakage por construção.
# =============================================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ── Configuração ─────────────────────────────────────────────
RANDOM_STATE = 42   # replicabilidade (exigência do desafio)
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data' / 'processed' / 'base_modelagem.parquet'

# ── Definição das features ───────────────────────────────────
# Numéricas: histórico, tendência e socioeconômicas
FEATURES_NUM = [
    'taxa_2023', 'taxa_2024', 'variacao_2023_2024', 'melhorou',
    'populacao', 'pib_per_capita'
]
# Categórica: estado (27 valores → one-hot)
FEATURES_CAT = ['uf']

# Colunas descartadas: id_municipio (identificador), regiao
# (redundante com uf), pib_total_mil (redundante com per capita),
# e o alvo.
ALVO = 'atingiu_meta_2025'


def carregar_dados():
    """Carrega a base e separa X (features) de y (alvo)."""
    df = pd.read_parquet(BASE)
    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[ALVO].copy()
    return X, y


def criar_preprocessador():
    """
    Monta o ColumnTransformer:
      - Numéricas: imputa nulos com a MEDIANA + padroniza (z-score)
      - Categórica: imputa nulos + one-hot encoding

    Retorna o ColumnTransformer (ainda não ajustado).
    """
    # Pipeline para variáveis numéricas
    transform_num = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para variáveis categóricas
    transform_cat = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combina os dois num ColumnTransformer
    preprocessador = ColumnTransformer(transformers=[
        ('num', transform_num, FEATURES_NUM),
        ('cat', transform_cat, FEATURES_CAT)
    ])

    return preprocessador


def separar_treino_teste(X, y, test_size=0.20):
    """
    Split estratificado pelo alvo (mantém a proporção 73/27).
    random_state fixo para replicabilidade.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,           # mantém proporção das classes
        random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


# ── Teste rápido quando executado diretamente ────────────────
if __name__ == '__main__':
    print('='*55)
    print('TESTE DO PIPELINE DE PRÉ-PROCESSAMENTO')
    print('='*55)

    X, y = carregar_dados()
    print(f'\nFeatures: {list(X.columns)}')
    print(f'Shape X: {X.shape}')
    print(f'Alvo — distribuição: {dict(y.value_counts())}')

    X_train, X_test, y_train, y_test = separar_treino_teste(X, y)
    print(f'\nTreino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}')
    print(f'Proporção alvo treino: {y_train.mean():.3f}')
    print(f'Proporção alvo teste:  {y_test.mean():.3f}')

    prep = criar_preprocessador()
    X_train_t = prep.fit_transform(X_train)
    X_test_t  = prep.transform(X_test)
    print(f'\nApós pré-processamento:')
    print(f'  X_train transformado: {X_train_t.shape}')
    print(f'  X_test transformado:  {X_test_t.shape}')
    print(f'  (colunas expandiram por causa do one-hot de uf)')
    print('\n✅ Pipeline funcionando — pronto para receber o modelo!')