# =============================================================
# TRAIN — Definição, treino e seleção de modelos
#
# Encapsula a lógica de modelagem: define os candidatos,
# avalia por validação cruzada, otimiza hiperparâmetros e
# treina o modelo campeão.
# =============================================================

import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV
)

# Importa o pré-processamento do módulo irmão
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'preprocessing'))
from pipeline import (
    carregar_dados, criar_preprocessador,
    separar_treino_teste, RANDOM_STATE
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)

# Validação cruzada estratificada (5 folds)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def definir_modelos():
    """
    Retorna os algoritmos candidatos.

    Justificativa da escolha:
    - Regressão Logística: baseline linear interpretável
    - Random Forest: ensemble robusto, captura não-linearidades
    - Gradient Boosting: ensemble sequencial, geralmente superior
      em dados tabulares de porte médio

    class_weight='balanced' compensa o desbalanceamento 73/27.
    """
    return {
        'Regressão Logística': LogisticRegression(
            max_iter=1000, class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        )
    }


def criar_baseline():
    """
    Baseline: prever sempre a classe majoritária.
    É o mínimo que qualquer modelo precisa superar para ter valor.
    """
    return Pipeline([
        ('prep', criar_preprocessador()),
        ('clf', DummyClassifier(strategy='most_frequent'))
    ])


def montar_pipelines(modelos):
    """Envolve cada modelo num Pipeline com o pré-processamento."""
    return {
        nome: Pipeline([
            ('prep', criar_preprocessador()),
            ('clf', modelo)
        ])
        for nome, modelo in modelos.items()
    }


def avaliar_por_cv(pipelines, X_train, y_train):
    """
    Avalia cada pipeline por validação cruzada em 5 métricas.
    Retorna DataFrame comparativo.
    """
    metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    resultados = []

    for nome, pipe in pipelines.items():
        linha = {'modelo': nome}
        for m in metricas:
            scores = cross_val_score(pipe, X_train, y_train, cv=CV, scoring=m)
            linha[m] = scores.mean()
            linha[f'{m}_std'] = scores.std()
        resultados.append(linha)

    return pd.DataFrame(resultados).set_index('modelo')


def otimizar_hiperparametros(X_train, y_train):
    """
    GridSearchCV sobre o Gradient Boosting.

    A grade foca em parâmetros que controlam overfitting:
    - learning_rate menor = aprendizado mais cauteloso
    - max_depth menor = árvores mais rasas, menos memorização
    - n_estimators = número de árvores no ensemble
    """
    param_grid = {
        'clf__n_estimators': [100, 200, 300],
        'clf__learning_rate': [0.05, 0.1],
        'clf__max_depth': [2, 3, 4]
    }

    pipe = Pipeline([
        ('prep', criar_preprocessador()),
        ('clf', GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])

    grid = GridSearchCV(
        pipe, param_grid, cv=CV, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid


def salvar_modelo(modelo, nome='modelo_gradient_boosting.pkl'):
    """Serializa o pipeline completo (pré-processamento + modelo)."""
    caminho = MODELS / nome
    joblib.dump(modelo, caminho)
    return caminho


if __name__ == '__main__':
    print('='*55)
    print('TREINAMENTO DE MODELOS')
    print('='*55)

    X, y = carregar_dados()
    X_train, X_test, y_train, y_test = separar_treino_teste(X, y)
    print(f'Treino: {len(X_train):,} | Teste: {len(X_test):,}\n')

    # Baseline
    base = criar_baseline()
    score_base = cross_val_score(
        base, X_train, y_train, cv=CV, scoring='accuracy'
    ).mean()
    print(f'BASELINE (classe majoritária): {score_base:.3f}\n')

    # Comparação dos candidatos
    pipelines = montar_pipelines(definir_modelos())
    print('Avaliando modelos por validação cruzada...')
    df_result = avaliar_por_cv(pipelines, X_train, y_train)
    print('\n' + df_result[
        ['accuracy','precision','recall','f1','roc_auc']
    ].round(3).to_string())

    # Otimização
    print('\nOtimizando hiperparâmetros do Gradient Boosting...')
    grid = otimizar_hiperparametros(X_train, y_train)
    print(f'Melhores parâmetros: {grid.best_params_}')
    print(f'Melhor F1 (CV): {grid.best_score_:.3f}')

    # Salva o campeão
    caminho = salvar_modelo(grid.best_estimator_)
    print(f'\n✅ Modelo salvo em: {caminho}')