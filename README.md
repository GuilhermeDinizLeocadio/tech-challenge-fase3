# Tech Challenge Fase 3 — Predição de Alfabetização Municipal

**Instituição:** FIAP PosTech  
**Curso:** AI Scientist  
**Aluno:** Guilherme Diniz Leocadio  
**Fase:** 3 — Modelagem Supervisionada  
**Repositório Fase 2:** https://github.com/GuilhermeDinizLeocadio/tech-challenge-fase2

---

## 1. Contexto e Objetivo

O Compromisso Nacional Criança Alfabetizada (Decreto nº 11.556/2023) estabelece
a meta de alfabetização universal até 2030. Com base nos dados do INEP (AEEB 2025)
e das metas municipais oficiais, este projeto constrói um modelo supervisionado de
classificação binária que prevê se um município **atingirá ou não sua meta de
alfabetização pactuada**.

**Pergunta central:** Quais fatores territoriais, socioeconômicos e históricos
explicam o cumprimento (ou não) da meta de alfabetização municipal?

---

## 2. Definição do Alvo

| Item | Decisão |
|---|---|
| **Variável-alvo** | `atingiu_meta_2025` (binária) |
| **Classe 1** | Município com taxa INEP 2025 ≥ meta pactuada 2025 |
| **Classe 0** | Município com taxa INEP 2025 < meta pactuada 2025 |
| **Grão** | Um município por linha (rede pública total, `ID_TIPO_REDE=5`) |
| **N base** | ~5.342 municípios |
| **Balanceamento** | 72,8% classe 1 / 27,2% classe 0 (desbalanceamento moderado) |

**Por que este alvo?** Alinhamento direto com política pública, usa metas oficiais,
e as escalas INEP/Base dos Dados foram confirmadas como compatíveis
(correlação 0,69 entre taxa 2024 e 2025).

**Atenção — data leakage:** São proibidos como features todos os dados derivados
do resultado de 2025: `PC_ALUNO_ALFABETIZADO`, `VL_MEDIA_LP`, `PC_ALUNO_NIVEL_*`,
`meta_alfabetizacao_2025` e qualquer coluna calculada a partir delas.

---

## 3. Fontes de Dados

| Fonte | Arquivo | Uso |
|---|---|---|
| INEP AEEB 2025 | `TS_MUNICIPIO.csv` | Taxa de alfabetização 2025 → **alvo** |
| Base dos Dados | `meta_municipio.csv` | Meta oficial 2025 → **alvo** |
| Base dos Dados | `municipio.csv` | Taxa histórica 2023/2024 → **feature** |
| IBGE | *a integrar* | População, PIB per capita → **features** |
| Atlas do IDH | *a integrar* | IDH municipal → **features** |
| FUNDEB | *a integrar* | Investimento por aluno → **features** |

---

## 4. Estrutura do Repositório

tech-challenge-fase3/
├── data/
│ ├── raw/ # CSVs brutos (não versionados)
│ └── processed/ # Base final de modelagem (não versionada)
├── notebooks/
│ ├── 01_eda.ipynb
│ └── 02_modeling.ipynb
├── src/
│ ├── preprocessing/ # build_base.py, pipeline.py
│ ├── modeling/ # train.py, tune.py
│ ├── evaluation/ # metrics.py, interpret.py
│ └── visualization/ # plots.py
├── models/ # Modelos serializados (não versionados)
├── reports/figures/ # Visualizações exportadas
├── images/
├── requirements.txt
├── README.md
└── .gitignore

---

## 5. Como Reproduzir

```bash
# 1. Clone o repositório
git clone https://github.com/GuilhermeDinizLeocadio/tech-challenge-fase3
cd tech-challenge-fase3

# 2. Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Coloque os CSVs brutos em data/raw/
# (TS_MUNICIPIO.csv, meta_municipio.csv, municipio.csv)

# 5. Construa a base de modelagem
python src/preprocessing/build_base.py

# 6. Execute os notebooks em ordem
# notebooks/01_eda.ipynb → notebooks/02_modeling.ipynb
```

---

## 6. Etapas do Projeto

- [x] Etapa 1 — Entendimento do problema e setup
- [ ] Etapa 2 — Análise Exploratória (EDA)
- [ ] Etapa 3 — Pipeline de ML (Scikit-learn)
- [ ] Etapa 4 — Treinamento e validação
- [ ] Etapa 5 — Avaliação e interpretabilidade
- [ ] Etapa 6 — Aplicação estratégica
- [ ] Etapa 7 — Entregáveis finais

---

## 7. Algoritmos Avaliados

*(a preencher na Etapa 4)*

---

## 8. Métricas de Avaliação

*(a preencher na Etapa 5)*

---

## 9. Principais Insights

*(a preencher na Etapa 6)*

---

## 10. Limitações e Evoluções Futuras

*(a preencher na Etapa 7)*

---

*FIAP PosTech — AI Scientist — Tech Challenge Fase 3*