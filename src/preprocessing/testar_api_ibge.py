# ============================================================
# TESTE — Validar acesso à API de Agregados do IBGE
# Roda 1 chamada pequena e mostra a estrutura da resposta
# ============================================================
import requests

HEADERS = {'User-Agent': 'Mozilla/5.0'}

# PIB per capita municipal — agregado 5938, variável 37, último período
url = ("https://servicodados.ibge.gov.br/api/v3/agregados/5938"
       "/periodos/-1/variaveis/37?localidades=N6[all]")

print('Consultando API do IBGE (PIB per capita municipal)...')
r = requests.get(url, headers=HEADERS, timeout=60)
print(f'Status HTTP: {r.status_code}')

if r.status_code == 200:
    data = r.json()
    series = data[0]['resultados'][0]['series']
    periodo = list(series[0]['serie'].keys())[0]
    print(f'✅ API respondeu!')
    print(f'   Variável: {data[0]["variavel"]}')
    print(f'   Unidade: {data[0]["unidade"]}')
    print(f'   Período mais recente: {periodo}')
    print(f'   Municípios retornados: {len(series):,}')
    print(f'\n   Amostra:')
    for s in series[:5]:
        loc = s['localidade']
        valor = s['serie'][periodo]
        print(f'     {loc["id"]} - {loc["nome"]}: R$ {valor}')
else:
    print(f'❌ Erro {r.status_code}: {r.text[:300]}')