# =============================================================
# RANKING COM NOMES — Municípios em risco e casos de sucesso
#
# Cruza as previsões do modelo com os nomes dos municípios
# e exporta rankings para uso em relatórios e apresentações.
# =============================================================

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'preprocessing'))
from pipeline import FEATURES_NUM, FEATURES_CAT
from metrics import carregar_modelo

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
REPORTS = ROOT / 'reports'


def carregar_nomes():
    """Extrai id_municipio e nome do arquivo do INEP."""
    df = pd.read_csv(RAW / 'TS_MUNICIPIO.csv', sep=';', encoding='latin-1')
    nomes = df[['CO_MUNICIPIO', 'NO_MUNICIPIO']].drop_duplicates('CO_MUNICIPIO')
    nomes.columns = ['id_municipio', 'municipio']
    nomes['id_municipio'] = nomes['id_municipio'].astype(int)
    return nomes


def casos_de_sucesso(df, n=10):
    """
    Municípios que REALMENTE cumpriram a meta de 2025,
    ordenados pela maior melhora de trajetória.
    Diferente de "menor risco previsto" — aqui é o
    resultado observado, não a previsão.
    """
    cumpriram = df[df['atingiu_meta_2025'] == 1].copy()
    return cumpriram.sort_values('variacao_2023_2024', ascending=False).head(n)


def main():
    print('=' * 70)
    print('RANKING DE MUNICÍPIOS — RISCO E CASOS DE SUCESSO')
    print('=' * 70)

    # Base completa
    df = pd.read_parquet(ROOT / 'data' / 'processed' / 'base_modelagem.parquet')
    X = df[FEATURES_NUM + FEATURES_CAT].copy()

    # Aplica o modelo
    modelo = carregar_modelo()
    df['prob_atingir'] = modelo.predict_proba(X)[:, 1].round(3)
    df['risco'] = (1 - df['prob_atingir']).round(3)

    # Junta os nomes
    nomes = carregar_nomes()
    df = df.merge(nomes, on='id_municipio', how='left')

    cols = ['municipio', 'uf', 'taxa_2024', 'variacao_2023_2024', 'risco']
    cols_sucesso = ['municipio', 'uf', 'taxa_2024', 'variacao_2023_2024']

    # 1. Maior risco previsto
    ranking = df.sort_values('risco', ascending=False)
    print('\n[1] TOP 15 — MAIOR RISCO PREVISTO PELO MODELO:')
    print(ranking[cols].head(15).to_string(index=False))

    # 2. Menor risco previsto
    print('\n\n[2] TOP 10 — MENOR RISCO PREVISTO:')
    print(ranking[cols].tail(10).sort_values('risco').to_string(index=False))

    # 3. Casos de sucesso REAIS (cumpriram a meta)
    print('\n\n[3] TOP 10 — CUMPRIRAM A META com maior melhora real:')
    print(casos_de_sucesso(df)[cols_sucesso].to_string(index=False))

    # 4. Quem NÃO cumpriu, com maior queda
    nao_cumpriram = df[df['atingiu_meta_2025'] == 0].copy()
    piores = nao_cumpriram.sort_values('variacao_2023_2024').head(10)
    print('\n\n[4] TOP 10 — NÃO CUMPRIRAM A META com maior queda real:')
    print(piores[cols_sucesso].to_string(index=False))

    # Exporta o ranking completo
    saida = REPORTS / 'ranking_municipios_risco.csv'
    export = ranking[['id_municipio', 'municipio', 'uf', 'taxa_2023',
                      'taxa_2024', 'variacao_2023_2024',
                      'atingiu_meta_2025', 'prob_atingir', 'risco']]
    export.to_csv(saida, index=False, encoding='utf-8-sig')
    print(f'\n💾 Ranking completo ({len(export):,} municípios) salvo em:')
    print(f'   {saida}')


if __name__ == '__main__':
    main()