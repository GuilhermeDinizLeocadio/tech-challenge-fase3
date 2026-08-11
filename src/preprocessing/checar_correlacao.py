# ============================================================
# CHECAR CORRELAÇÃO — Verifica sinal das novas features
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_parquet(ROOT / 'data' / 'processed' / 'base_modelagem.parquet')

# PIB per capita e população costumam ser muito assimétricos.
# Aplicamos log para a correlação fazer sentido.
df['log_populacao'] = np.log1p(df['populacao'])
df['log_pib_pc']    = np.log1p(df['pib_per_capita'])

features = ['taxa_2023', 'taxa_2024', 'variacao_2023_2024', 'melhorou',
            'populacao', 'pib_per_capita', 'log_populacao', 'log_pib_pc']

print('CORRELAÇÃO DAS FEATURES COM O ALVO (atingiu_meta_2025)')
print('='*55)
corr = df[features + ['atingiu_meta_2025']].corr()['atingiu_meta_2025']
corr = corr.drop('atingiu_meta_2025').sort_values(key=abs, ascending=False)
for feat, val in corr.items():
    sinal = '+' if val >= 0 else '-'
    print(f'   {feat:18s}: {sinal}{abs(val):.3f}')

print('\nCORRELAÇÃO ENTRE AS FEATURES (evitar redundância)')
print('='*55)
print(df[['taxa_2024','log_pib_pc','log_populacao']].corr().round(2).to_string())