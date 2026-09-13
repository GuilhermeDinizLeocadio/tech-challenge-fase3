# ============================================================
# ENRIQUECER IBGE — Adiciona população e PIB per capita
#
# Busca na API de Agregados do IBGE:
#   - População estimada (agregado 6579, variável 9324)
#   - PIB total municipal (agregado 5938, variável 37)
# Calcula PIB per capita = PIB / população
# Integra à base_modelagem.parquet pela chave id_municipio
# ============================================================
import requests
import pandas as pd
from pathlib import Path

HEADERS = {'User-Agent': 'Mozilla/5.0'}
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data' / 'processed' / 'base_modelagem.parquet'
RAW  = ROOT / 'data' / 'raw'

def buscar_agregado(agregado, variavel, descricao):
    """
    Busca uma variável da API de Agregados do IBGE para
    todos os municípios (N6), no período mais recente (-1).
    Retorna DataFrame com id_municipio e o valor.
    """
    print(f'📥 Buscando {descricao}...')
    url = (f'https://servicodados.ibge.gov.br/api/v3/agregados/{agregado}'
           f'/periodos/-1/variaveis/{variavel}?localidades=N6[all]')
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()

    data = r.json()
    series = data[0]['resultados'][0]['series']
    periodo = list(series[0]['serie'].keys())[0]

    registros = []
    for s in series:
        id_mun = int(s['localidade']['id'])
        valor_raw = s['serie'][periodo]
        # Valores ausentes vêm como '-' ou '...'
        try:
            valor = float(valor_raw)
        except (ValueError, TypeError):
            valor = None
        registros.append({'id_municipio': id_mun, 'valor': valor})

    df = pd.DataFrame(registros)
    print(f'   ✅ {len(df):,} municípios (período {periodo})')
    return df

def main():
    print('='*55)
    print('ENRIQUECIMENTO IBGE — População e PIB')
    print('='*55)

    # 1. Carrega base atual
    df = pd.read_parquet(BASE)
    print(f'\nBase atual: {df.shape[0]:,} linhas x {df.shape[1]} colunas')

    # 2. Busca população estimada (agregado 6579, variável 9324)
    pop = buscar_agregado(6579, 9324, 'População estimada')
    pop.columns = ['id_municipio', 'populacao']

    # 3. Busca PIB total (agregado 5938, variável 37) — em Mil Reais
    pib = buscar_agregado(5938, 37, 'PIB total municipal')
    pib.columns = ['id_municipio', 'pib_total_mil']

    # 4. Integra à base
    print('\n🔗 Integrando à base...')
    df = df.merge(pop, on='id_municipio', how='left')
    df = df.merge(pib, on='id_municipio', how='left')

    # 5. Calcula PIB per capita (R$ por habitante)
    # pib_total está em Mil Reais → multiplica por 1000
    df['pib_per_capita'] = (
        (df['pib_total_mil'] * 1000) / df['populacao']
    ).round(2)

    # 6. Diagnóstico de nulos das novas colunas
    print('\n📊 NULOS NAS NOVAS FEATURES:')
    for col in ['populacao', 'pib_total_mil', 'pib_per_capita']:
        n = df[col].isnull().sum()
        print(f'   {col}: {n} nulos ({n/len(df)*100:.1f}%)')

    # 7. Salva a base enriquecida
    df.to_parquet(BASE, index=False, engine='pyarrow')
    print(f'\n💾 Base enriquecida salva: {df.shape[0]:,} x {df.shape[1]} colunas')
    print(f'   Colunas: {list(df.columns)}')
    print('✅ Enriquecimento IBGE concluído!')

if __name__ == '__main__':
    main()