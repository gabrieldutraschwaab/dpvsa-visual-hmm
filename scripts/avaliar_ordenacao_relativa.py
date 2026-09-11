#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]

ENTRADA = (
    RAIZ
    / "resultados/validacao_independente"
    / "comparacao_validacao_p10.csv"
)
SAIDA_DETALHADA = (
    RAIZ
    / "resultados/validacao_independente"
    / "ordenacao_relativa.csv"
)
SAIDA_RESUMO = (
    RAIZ
    / "resultados/validacao_independente"
    / "resumo_ordenacao_relativa.csv"
)

METODOS = {
    "centro_api": "distancia_centro_api_m",
    "obb_p10": "distancia_obb_p10_m",
}

dados = pd.read_csv(ENTRADA)

colunas_necessarias = {
    "imagem",
    "posicao_relativa",
    "deteccao",
    "classe",
    "distancia_real_m",
    *METODOS.values(),
}

colunas_ausentes = sorted(
    colunas_necessarias - set(dados.columns)
)

if colunas_ausentes:
    raise ValueError(
        f"Colunas ausentes: {colunas_ausentes}"
    )

if dados[
    ["imagem", "posicao_relativa"]
].duplicated().any():
    raise ValueError(
        "Há posições relativas duplicadas na mesma imagem."
    )

registros = []

for imagem, grupo in dados.groupby("imagem", sort=True):
    posicoes = set(grupo["posicao_relativa"])

    if len(grupo) != 2 or posicoes != {
        "mais_proxima",
        "mais_distante",
    }:
        raise ValueError(
            f"{imagem} não possui exatamente um par "
            "próxima/distante."
        )

    por_posicao = grupo.set_index("posicao_relativa")
    proxima = por_posicao.loc["mais_proxima"]
    distante = por_posicao.loc["mais_distante"]

    margem_real = (
        distante["distancia_real_m"]
        - proxima["distancia_real_m"]
    )

    if margem_real <= 0:
        raise ValueError(
            f"Margem real inválida em {imagem}: "
            f"{margem_real}"
        )

    for metodo, coluna in METODOS.items():
        estimativa_proxima = proxima[coluna]
        estimativa_distante = distante[coluna]
        margem_estimada = (
            estimativa_distante - estimativa_proxima
        )
        erro_margem = margem_estimada - margem_real

        registros.append(
            {
                "imagem": imagem,
                "metodo": metodo,
                "deteccao_proxima": int(
                    proxima["deteccao"]
                ),
                "deteccao_distante": int(
                    distante["deteccao"]
                ),
                "classe_proxima": proxima["classe"],
                "classe_distante": distante["classe"],
                "mesma_classe": (
                    proxima["classe"]
                    == distante["classe"]
                ),
                "distancia_real_proxima_m": (
                    proxima["distancia_real_m"]
                ),
                "distancia_real_distante_m": (
                    distante["distancia_real_m"]
                ),
                "margem_real_m": margem_real,
                "distancia_estimada_proxima_m": (
                    estimativa_proxima
                ),
                "distancia_estimada_distante_m": (
                    estimativa_distante
                ),
                "margem_estimada_m": margem_estimada,
                "ordenacao_correta": (
                    estimativa_proxima
                    < estimativa_distante
                ),
                "erro_assinado_margem_m": erro_margem,
                "erro_absoluto_margem_m": abs(
                    erro_margem
                ),
                "erro_percentual_absoluto_margem": (
                    100
                    * abs(erro_margem)
                    / margem_real
                ),
            }
        )

resultado = pd.DataFrame(registros)

resumos = []

for metodo, grupo in resultado.groupby(
    "metodo",
    sort=False,
):
    erros = grupo["erro_assinado_margem_m"]
    pares_mesma_classe = grupo.loc[
        grupo["mesma_classe"]
    ]

    resumos.append(
        {
            "metodo": metodo,
            "quantidade_pares": len(grupo),
            "ordenacoes_corretas": int(
                grupo["ordenacao_correta"].sum()
            ),
            "acuracia_ordenacao_percentual": (
                100
                * grupo["ordenacao_correta"].mean()
            ),
            "vies_margem_m": erros.mean(),
            "mae_margem_m": erros.abs().mean(),
            "rmse_margem_m": np.sqrt(
                np.mean(np.square(erros))
            ),
            "quantidade_pares_mesma_classe": len(
                pares_mesma_classe
            ),
            "ordenacoes_corretas_mesma_classe": int(
                pares_mesma_classe[
                    "ordenacao_correta"
                ].sum()
            ),
            "acuracia_mesma_classe_percentual": (
                100
                * pares_mesma_classe[
                    "ordenacao_correta"
                ].mean()
                if len(pares_mesma_classe)
                else np.nan
            ),
        }
    )

resumo = pd.DataFrame(resumos)

SAIDA_DETALHADA.parent.mkdir(
    parents=True,
    exist_ok=True,
)

resultado.to_csv(
    SAIDA_DETALHADA,
    index=False,
    float_format="%.6f",
)
resumo.to_csv(
    SAIDA_RESUMO,
    index=False,
    float_format="%.6f",
)

print("\nOrdenação por imagem e método:\n")
print(
    resultado[
        [
            "imagem",
            "metodo",
            "mesma_classe",
            "margem_real_m",
            "margem_estimada_m",
            "erro_absoluto_margem_m",
            "ordenacao_correta",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print("\nResumo da ordenação relativa:\n")
print(
    resumo.round(4).to_string(index=False)
)

print(f"\nDetalhamento salvo em: {SAIDA_DETALHADA}")
print(f"Resumo salvo em: {SAIDA_RESUMO}")
