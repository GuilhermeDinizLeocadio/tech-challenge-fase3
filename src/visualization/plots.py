# =============================================================
# PLOTS — Geração de visualizações do projeto
#
# Centraliza os gráficos usados nos relatórios e no README.
# Todas as figuras são salvas em reports/figures/.
# =============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # backend sem interface gráfica
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / 'reports' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11


def plot_distribuicao_alvo(df, coluna_alvo='atingiu_meta_2025',
                           nome='dist_alvo.png'):
    """Gráfico de barras da distribuição das classes do alvo."""
    fig, ax = plt.subplots(figsize=(6, 5))
    valores = df[coluna_alvo].value_counts().sort_index()
    total = valores.sum()
    labels = [f'Não atingiu (0)\n{valores[0]/total:.1%}',
              f'Atingiu (1)\n{valores[1]/total:.1%}']

    ax.bar(labels, valores, color=['#e74c3c', '#2ecc71'],
           width=0.5, edgecolor='white')
    ax.set_title('Distribuição do Alvo', fontweight='bold', pad=15)
    ax.set_ylabel('Nº de Municípios')
    for i, v in enumerate(valores):
        ax.text(i, v + total*0.01, f'{v:,}', ha='center', fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGURES / nome, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return FIGURES / nome


def plot_comparacao_modelos(df_result, baseline,
                            nome='comparacao_modelos.png'):
    """Barras agrupadas comparando modelos em várias métricas."""
    metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(metricas))
    largura = 0.25
    cores = ['#3498db', '#e67e22', '#2ecc71']

    for i, nome_modelo in enumerate(df_result.index):
        valores = df_result.loc[nome_modelo, metricas].values.astype(float)
        ax.bar(x + i*largura, valores, largura,
               label=nome_modelo, color=cores[i % len(cores)])

    ax.axhline(baseline, color='red', linestyle='--', linewidth=1.5,
               label=f'Baseline ({baseline:.3f})')
    ax.set_xticks(x + largura)
    ax.set_xticklabels(['Accuracy','Precision','Recall','F1','ROC-AUC'])
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.0)
    ax.set_title('Comparação dos Modelos — Validação Cruzada 5-fold',
                 fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES / nome, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return FIGURES / nome


def plot_feature_importance(df_imp, top=12,
                            nome='feature_importance.png'):
    """Barras horizontais com as features mais importantes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    dados = df_imp.head(top).sort_values('importancia')

    ax.barh(dados['feature'], dados['importancia'],
            color='#2ecc71', edgecolor='white')
    ax.set_xlabel('Importância')
    ax.set_title(f'Feature Importance — Top {top}',
                 fontweight='bold', pad=15)

    plt.tight_layout()
    fig.savefig(FIGURES / nome, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return FIGURES / nome


def plot_matriz_confusao(cm_df, nome='matriz_confusao.png'):
    """Heatmap da matriz de confusão."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                cbar=False, ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title('Matriz de Confusão — Conjunto de Teste',
                 fontweight='bold', pad=15)

    plt.tight_layout()
    fig.savefig(FIGURES / nome, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return FIGURES / nome


def plot_threshold(df_thr, nome='threshold_analysis.png'):
    """Linha mostrando o trade-off precision/recall por threshold."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(df_thr['threshold'], df_thr['recall_risco'],
            marker='o', linewidth=2, color='#e74c3c',
            label='Recall (risco capturado)')
    ax.plot(df_thr['threshold'], df_thr['precision_risco'],
            marker='s', linewidth=2, color='#3498db',
            label='Precision (acerto do alerta)')
    ax.axvline(0.5, color='gray', linestyle='--',
               label='Threshold padrão (0.5)')

    ax.set_xlabel('Threshold de decisão')
    ax.set_ylabel('Score')
    ax.set_title('Trade-off de Threshold — Classe "Em Risco"',
                 fontweight='bold', pad=15)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / nome, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return FIGURES / nome


if __name__ == '__main__':
    # Demonstração: gera a matriz de confusão e o gráfico de threshold
    sys.path.insert(0, str(ROOT / 'src' / 'evaluation'))
    sys.path.insert(0, str(ROOT / 'src' / 'preprocessing'))
    from metrics import (
        carregar_modelo, matriz_confusao,
        feature_importance, analisar_threshold
    )
    from pipeline import carregar_dados, separar_treino_teste

    print('Gerando visualizações...')
    modelo = carregar_modelo()
    X, y = carregar_dados()
    _, X_test, _, y_test = separar_treino_teste(X, y)

    p1 = plot_matriz_confusao(matriz_confusao(modelo, X_test, y_test))
    print(f'  ✅ {p1.name}')

    p2 = plot_feature_importance(feature_importance(modelo))
    print(f'  ✅ {p2.name}')

    p3 = plot_threshold(analisar_threshold(modelo, X_test, y_test))
    print(f'  ✅ {p3.name}')

    print('\n✅ Visualizações geradas em reports/figures/')