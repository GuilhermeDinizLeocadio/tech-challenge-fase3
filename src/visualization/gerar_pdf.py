# =============================================================
# GERAR PDF — Documentação técnica completa
#
# Converte o README.md em PDF e acrescenta um ANEXO com
# todas as figuras geradas no projeto.
#
# Saída: reports/documentacao_tecnica.pdf
# =============================================================

from pathlib import Path
import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / 'README.md'
FIGURES = ROOT / 'reports' / 'figures'
SAIDA = ROOT / 'reports' / 'documentacao_tecnica.pdf'

# Metadados do documento
META_TITULO = 'Tech Challenge Fase 3 — Predição de Alfabetização Municipal'
META_AUTOR = 'Guilherme Diniz Leocadio'
META_ASSUNTO = ('Modelagem supervisionada para predição de cumprimento '
                'de metas de alfabetização municipal')
META_KEYWORDS = ('machine learning, alfabetização, políticas públicas, '
                 'FIAP, scikit-learn')

# Figuras do anexo: (arquivo, título, legenda)
ANEXO_FIGURAS = [
    ('01_distribuicoes_gerais.png',
     'A.1 — Distribuições Gerais da Base',
     'Distribuição do alvo e das taxas de alfabetização de 2023 e 2024. '
     'As taxas seguem distribuição aproximadamente normal, com deslocamento '
     'positivo de 2023 para 2024.'),

    ('02_analise_regional.png',
     'A.2 — Análise Regional: Meta vs Taxa Absoluta',
     'Comparação entre o percentual de municípios que atingiram a meta e a '
     'taxa média absoluta por região. Evidencia o paradoxo: regiões com taxa '
     'alta não são necessariamente as que mais cumprem metas.'),

    ('03_boxplot_por_classe.png',
     'A.3 — Separação das Classes pelo Histórico',
     'Boxplots das taxas de 2023 e 2024 por classe do alvo. A taxa de 2024 '
     'separa bem as classes; a de 2023 apresenta relação invertida, '
     'refletindo o efeito das metas mais ambiciosas.'),

    ('04_correlacoes.png',
     'A.4 — Matriz de Correlação',
     'Correlações entre as variáveis numéricas e com o alvo. Confirma a '
     'correlação positiva da taxa de 2024 (+0,209) e negativa da taxa de '
     '2023 (−0,092).'),

    ('05_comparacao_modelos.png',
     'A.5 — Comparação dos Modelos',
     'Desempenho dos três algoritmos candidatos em cinco métricas, via '
     'validação cruzada 5-fold. A linha tracejada indica o baseline.'),

    ('06_feature_importance.png',
     'A.6 — Feature Importance',
     'Importância relativa das features no modelo Gradient Boosting. '
     'A distribuição equilibrada corrobora a ausência de data leakage.'),

    ('07_shap_summary.png',
     'A.7 — SHAP Summary Plot',
     'Impacto e direção de cada feature na previsão. Pontos à direita '
     'empurram para "atingiu a meta"; à esquerda, para "não atingiu". '
     'A cor indica o valor da feature (vermelho = alto).'),

    ('08_shap_waterfall.png',
     'A.8 — SHAP Waterfall (Explicação Individual)',
     'Decomposição da previsão de um município específico classificado como '
     'em risco. Mostra como cada feature contribuiu para a decisão final, '
     'partindo do valor esperado médio.'),

    ('09_threshold_analysis.png',
     'A.9 — Curva Precision-Recall e Threshold',
     'Trade-off entre precision e recall para a classe "em risco" em '
     'diferentes limiares de decisão.'),

    ('10_perfil_regional.png',
     'A.10 — Perfil de Risco por Estado',
     'Posicionamento dos estados segundo taxa média de 2024 (eixo X) e '
     'variação 2023→2024 (eixo Y). O tamanho da bolha representa o risco '
     'médio. O Rio Grande do Sul destaca-se como outlier.'),

    ('matriz_confusao.png',
     'A.11 — Matriz de Confusão',
     'Distribuição dos acertos e erros no conjunto de teste. Os 170 falsos '
     'negativos representam municípios em risco real não sinalizados '
     'pelo modelo no limiar padrão.'),
]

CSS_ESTILO = """
@page {
    size: a4 portrait;
    margin: 2cm 1.8cm;
    @frame footer {
        -pdf-frame-content: rodape;
        bottom: 1cm;
        margin-left: 1.8cm;
        margin-right: 1.8cm;
        height: 1cm;
    }
}
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.4;
    color: #222222;
}
h1 {
    color: #1a4d7a;
    font-size: 19pt;
    border-bottom: 2px solid #1a4d7a;
    padding-bottom: 6px;
}
h2 {
    color: #1a4d7a;
    font-size: 13.5pt;
    border-bottom: 1px solid #cccccc;
    padding-bottom: 3px;
    margin-top: 18px;
    -pdf-keep-with-next: true;
}
h3 {
    color: #2c6ca0;
    font-size: 11pt;
    margin-top: 12px;
    -pdf-keep-with-next: true;
}
table {
    width: 100%;
    margin: 10px 0;
    font-size: 8.5pt;
    -pdf-keep-in-frame-mode: shrink;
}
th {
    background-color: #1a4d7a;
    color: #ffffff;
    padding: 5px;
    text-align: left;
    border: 1px solid #1a4d7a;
}
td {
    border: 1px solid #dddddd;
    padding: 4px 6px;
}
code {
    background-color: #f0f2f5;
    font-family: Courier, monospace;
    font-size: 8.5pt;
}
pre {
    background-color: #f7f9fb;
    border-left: 3px solid #1a4d7a;
    padding: 8px;
    font-family: Courier, monospace;
    font-size: 7pt;
}
blockquote {
    border-left: 3px solid #f39c12;
    background-color: #fff8e7;
    padding: 6px 10px;
    margin: 10px 0;
}
.legenda {
    font-size: 8.5pt;
    color: #555555;
    font-style: italic;
    margin-bottom: 14px;
}
.figura {
    -pdf-keep-in-frame-mode: shrink;
.figura img {
    width: 17cm;
    margin: 14px 0;
}
.quebra {
    page-break-before: always;
}
#rodape {
    font-size: 8pt;
    color: #888888;
    text-align: center;
}
"""


def montar_anexo():
    """
    Monta o HTML do anexo. Cada figura ocupa sua própria página,
    com largura ampliada para melhor legibilidade.
    """
    partes = [
        '<div class="quebra">',
        '<h1>Anexo — Figuras do Projeto</h1>',
        '<p>Este anexo reúne todas as visualizações geradas ao longo do '
        'projeto, incluindo as que não foram incorporadas ao README para '
        'mantê-lo conciso.</p>',
        '</div>',
    ]

    incluidas = 0
    for arquivo, titulo, legenda in ANEXO_FIGURAS:
        caminho = FIGURES / arquivo
        if not caminho.exists():
            print(f'   ⚠️  Figura não encontrada, ignorada: {arquivo}')
            continue

        partes.append('<div class="figura quebra">')
        partes.append(f'<h2>{titulo}</h2>')
        partes.append(f'<img src="reports/figures/{arquivo}">')
        partes.append(f'<p class="legenda">{legenda}</p>')
        partes.append('</div>')
        incluidas += 1

    print(f'   📊 {incluidas} figuras incluídas no anexo')
    return '\n'.join(partes)


def main():
    print('📄 Gerando documentação técnica em PDF...')

    texto = README.read_text(encoding='utf-8')
    html_readme = markdown.markdown(
        texto, extensions=['tables', 'fenced_code']
    )

    html_anexo = montar_anexo()

    html_completo = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>{META_TITULO}</title>
        <meta name="author" content="{META_AUTOR}">
        <meta name="subject" content="{META_ASSUNTO}">
        <meta name="keywords" content="{META_KEYWORDS}">
        <style>{CSS_ESTILO}</style>
      </head>
      <body>
        <div id="rodape">
          Tech Challenge Fase 3 — FIAP PosTech | Página <pdf:pagenumber>
        </div>
        {html_readme}
        {html_anexo}
      </body>
    </html>
    """

    with open(SAIDA, 'wb') as f:
        status = pisa.CreatePDF(
            html_completo,
            dest=f,
            encoding='utf-8',
            path=str(ROOT) + '/'
        )

    if status.err:
        print(f'⚠️  Gerado com {status.err} avisos')
    else:
        print('✅ PDF gerado sem erros')

    tamanho = SAIDA.stat().st_size / 1024
    print(f'   Arquivo: {SAIDA}')
    print(f'   Tamanho: {tamanho:.0f} KB')


if __name__ == '__main__':
    main()