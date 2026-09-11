# =============================================================
# METRICS — Avaliação e interpretabilidade do modelo
#
# Calcula as métricas de classificação no conjunto de teste,
# extrai feature importance e analisa o trade-off de threshold.
# =============================================================

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'preprocessing'))
from pipeline import carregar_dados, separar_treino_teste

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'models'
REPORTS = ROOT / 'reports'


def carregar_modelo(nome='modelo_gradient_boosting.pkl'):
    """Carrega o pipeline serializado (pré-processamento + modelo)."""
    return joblib.load(MODELS / nome)


def avaliar(modelo, X_test, y_test):
    """
    Calcula as métricas de classificação no conjunto de teste.

    Retorna dicionário com as métricas globais e o recall
    específico da classe 0 (municípios em risco), que é a
    classe de maior interesse para política pública.
    """
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    return {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 3),
        'precision': round(float(precision_score(y_test, y_pred)), 3),
        'recall':    round(float(recall_score(y_test, y_pred)), 3),
        'f1':        round(float(f1_score(y_test, y_pred)), 3),
        'roc_auc':   round(float(roc_auc_score(y_test, y_proba)), 3),
        'recall_classe_0': round(float(
            recall_score(y_test, y_pred, pos_label=0)), 3),
    }


def matriz_confusao(modelo, X_test, y_test):
    """Retorna a matriz de confusão como DataFrame legível."""
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return pd.DataFrame(
        cm,
        index=['Real: Não atingiu', 'Real: Atingiu'],
        columns=['Prev: Não atingiu', 'Prev: Atingiu']
    )


def feature_importance(modelo):
    """
    Extrai a importância das features do modelo treinado,
    recuperando os nomes reais após o one-hot encoding.
    """
    prep = modelo.named_steps['prep']
    clf = modelo.named_steps['clf']

    nomes = prep.get_feature_names_out()
    nomes = [n.replace('num__', '').replace('cat__', '') for n in nomes]

    df = pd.DataFrame({
        'feature': nomes,
        'importancia': clf.feature_importances_
    }).sort_values('importancia', ascending=False)

    return df.reset_index(drop=True)


def analisar_threshold(modelo, X_test, y_test, thresholds=None):
    """
    Analisa o trade-off precision/recall para a classe 0
    (municípios em risco) em diferentes limiares de decisão.

    O limiar padrão (0.5) é arbitrário. Para política pública,
    pode ser preferível um limiar menor, capturando mais
    municípios em risco ao custo de mais falsos alarmes.
    """
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.60]

    y_risco = (y_test == 0).astype(int)
    prob_risco = 1 - modelo.predict_proba(X_test)[:, 1]

    linhas = []
    for t in thresholds:
        pred = (prob_risco >= t).astype(int)
        linhas.append({
            'threshold': t,
            'recall_risco': round(float(
                recall_score(y_risco, pred, zero_division=0)), 3),
            'precision_risco': round(float(
                precision_score(y_risco, pred, zero_division=0)), 3),
        })

    return pd.DataFrame(linhas)


def salvar_metricas(metricas, extras=None, nome='metricas_finais.json'):
    """Persiste as métricas em JSON para rastreabilidade."""
    saida = {'modelo': 'Gradient Boosting (otimizado)', 'teste': metricas}
    if extras:
        saida.update(extras)

    caminho = REPORTS / nome
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    return caminho


if __name__ == '__main__':
    print('='*55)
    print('AVALIAÇÃO DO MODELO')
    print('='*55)

    modelo = carregar_modelo()
    X, y = carregar_dados()
    X_train, X_test, y_train, y_test = separar_treino_teste(X, y)

    # Métricas no teste
    m = avaliar(modelo, X_test, y_test)
    print(f'\nConjunto de teste ({len(X_test):,} municípios):')
    for k, v in m.items():
        print(f'  {k}: {v}')

    # Matriz de confusão
    print('\nMatriz de confusão:')
    print(matriz_confusao(modelo, X_test, y_test).to_string())

    # Feature importance
    print('\nTop 10 features:')
    print(feature_importance(modelo).head(10).to_string(index=False))

    # Threshold
    print('\nTrade-off de threshold (classe "em risco"):')
    print(analisar_threshold(modelo, X_test, y_test).to_string(index=False))

    print('\n✅ Avaliação concluída.')