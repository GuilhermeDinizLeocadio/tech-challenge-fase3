# =============================================================
# BUILD BASE — Construção da Base de Modelagem
#
# Lê os CSVs Bronze locais, aplica as regras de negócio
# definidas na Etapa 1, e salva a base de modelagem em
# data/processed/base_modelagem.parquet
#
# ALVO: atingiu_meta_2025
#   1 = município com taxa INEP 2025 >= meta pactuada 2025
#   0 = município com taxa INEP 2025 <  meta pactuada 2025
#
# GRÃO: um município por linha (rede pública total, tipo 5)
# =============================================================

import pandas as pd
from pathlib import Path

# ── Caminhos ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW  = ROOT / 'data' / 'raw'
OUT  = ROOT / 'data' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)

# ── Mapeamento de regiões ────────────────────────────────────
REGIOES = {
    'AC':'Norte','AM':'Norte','AP':'Norte','PA':'Norte',
    'RO':'Norte','RR':'Norte','TO':'Norte',
    'AL':'Nordeste','BA':'Nordeste','CE':'Nordeste','MA':'Nordeste',
    'PB':'Nordeste','PE':'Nordeste','PI':'Nordeste',
    'RN':'Nordeste','SE':'Nordeste',
    'DF':'Centro-Oeste','GO':'Centro-Oeste',
    'MS':'Centro-Oeste','MT':'Centro-Oeste',
    'ES':'Sudeste','MG':'Sudeste','RJ':'Sudeste','SP':'Sudeste',
    'PR':'Sul','RS':'Sul','SC':'Sul'
}

def carregar_inep():
    """
    Lê TS_MUNICIPIO.csv (INEP 2025).
    Filtra rede pública total (ID_TIPO_REDE == 5).
    Retorna uma linha por município com a taxa de 2025.
    """
    print('📥 Carregando INEP 2025...')
    df = pd.read_csv(
        RAW / 'TS_MUNICIPIO.csv',
        sep=';',
        encoding='latin-1'
    )
    # Filtro de rede: 5 = pública total
    df5 = df[df['ID_TIPO_REDE'] == 5].copy()

    # Seleciona só o que interessa
    df5 = df5[['CO_MUNICIPIO', 'SG_UF', 'PC_ALUNO_ALFABETIZADO']].copy()
    df5.columns = ['id_municipio', 'uf', 'taxa_2025']
    df5['id_municipio'] = df5['id_municipio'].astype(int)

    print(f'   ✅ {len(df5):,} municípios (rede 5)')
    return df5

def carregar_meta():
    """
    Lê meta_municipio.csv (Base dos Dados).
    Retorna a meta pactuada para 2025 por município.
    """
    print('📥 Carregando metas municipais...')
    df = pd.read_csv(
        RAW / 'meta_municipio.csv',
        sep=',',
        encoding='utf-8'
    )
    # Pega só a meta de 2025, uma linha por município
    meta = df[['id_municipio', 'meta_alfabetizacao_2025']].drop_duplicates(
        'id_municipio'
    )
    meta['id_municipio'] = meta['id_municipio'].astype(int)
    print(f'   ✅ {len(meta):,} metas carregadas')
    return meta

def carregar_historico():
    """
    Lê municipio.csv (Base dos Dados, anos 2023 e 2024).
    Retorna taxa histórica pivotada: uma linha por município,
    colunas taxa_2023 e taxa_2024.
    IMPORTANTE: esses dados são de ANTES de 2025,
    portanto são features legítimas (não são leakage).
    """
    print('📥 Carregando histórico 2023-2024...')
    df = pd.read_csv(
        RAW / 'municipio.csv',
        sep=',',
        encoding='utf-8'
    )
    # Filtra rede 5 e anos de interesse
    df5 = df[(df['rede'] == 5) & (df['ano'].isin([2023, 2024]))].copy()
    df5 = df5[['id_municipio', 'ano', 'taxa_alfabetizacao']]
    df5['id_municipio'] = df5['id_municipio'].astype(int)

    # Pivota: uma linha por município, colunas por ano
    pivot = df5.pivot_table(
        index='id_municipio',
        columns='ano',
        values='taxa_alfabetizacao',
        aggfunc='mean'
    ).reset_index()
    pivot.columns = ['id_municipio', 'taxa_2023', 'taxa_2024']

    print(f'   ✅ {len(pivot):,} municípios com histórico')
    return pivot

def construir_alvo(df):
    """
    Cria a variável-alvo binária:
      1 = atingiu a meta de 2025
      0 = não atingiu

    ATENÇÃO: taxa_2025 e meta_2025 são usadas APENAS
    para calcular o alvo e depois descartadas como features.
    """
    df['atingiu_meta_2025'] = (
        df['taxa_2025'] >= df['meta_alfabetizacao_2025']
    ).astype(int)
    return df

def adicionar_features_territoriais(df):
    """
    Adiciona região geográfica a partir da UF.
    É a única feature territorial disponível antes
    do enriquecimento externo (Semana 2).
    """
    df['regiao'] = df['uf'].map(REGIOES)
    return df

def descartar_leakage(df):
    """
    Remove colunas que são o alvo disfarçado.
    Documenta explicitamente o que foi descartado e por quê.
    """
    colunas_leakage = [
        'taxa_2025',              # é o desfecho — define o alvo
        'meta_alfabetizacao_2025' # é o threshold do alvo
    ]
    df = df.drop(columns=colunas_leakage)
    print(f'   🚫 Leakage descartado: {colunas_leakage}')
    return df

def diagnostico(df):
    """Imprime o raio-x da base construída."""
    print('\n' + '='*55)
    print('DIAGNÓSTICO DA BASE DE MODELAGEM')
    print('='*55)
    print(f'Shape: {df.shape[0]:,} linhas x {df.shape[1]} colunas')
    print(f'\nColunas: {list(df.columns)}')
    print(f'\nAlvo (atingiu_meta_2025):')
    vc = df['atingiu_meta_2025'].value_counts()
    for v, c in vc.items():
        label = 'Atingiu   (1)' if v == 1 else 'Não atingiu (0)'
        print(f'  {label}: {c:,} ({c/len(df)*100:.1f}%)')
    print(f'\nNulos por coluna:')
    nulos = df.isnull().sum()
    for col, n in nulos.items():
        status = '✅' if n == 0 else '⚠️ '
        print(f'  {status} {col}: {n}')
    print('='*55)

def main():
    print('='*55)
    print('BUILD BASE — Tech Challenge Fase 3')
    print('='*55)

    # 1. Carregar fontes
    inep  = carregar_inep()
    meta  = carregar_meta()
    hist  = carregar_historico()

    # 2. Juntar tudo pelo id_municipio
    print('\n🔗 Juntando fontes...')
    df = inep.merge(meta,  on='id_municipio', how='inner')
    df = df.merge(hist,    on='id_municipio', how='left')
    print(f'   ✅ {len(df):,} municípios após joins')

    # 3. Construir alvo
    print('\n🎯 Construindo alvo...')
    df = construir_alvo(df)

    # 4. Features territoriais
    print('\n🗺️  Adicionando features territoriais...')
    df = adicionar_features_territoriais(df)

    # 5. Descartar leakage
    print('\n🚫 Descartando leakage...')
    df = descartar_leakage(df)

    # 6. Diagnóstico
    diagnostico(df)

    # 7. Salvar
    saida = OUT / 'base_modelagem.parquet'
    df.to_parquet(saida, index=False, engine='pyarrow')
    print(f'\n💾 Salvo em: {saida}')
    print('✅ Build concluído!')

if __name__ == '__main__':
    main()