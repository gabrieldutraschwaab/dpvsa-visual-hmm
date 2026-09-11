#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]

CAMINHO_PREVISOES = (
    RAIZ / "dados/processados/previsoes_validacao_p10.csv"
)
CAMINHO_GABARITO = (
    RAIZ / "dados/medidas_reais/validacao_independente.csv"
)
PASTA_RESULTADOS = (
    RAIZ / "resultados/validacao_independente"
)

CAMINHO_COMPARACAO = (
    PASTA_RESULTADOS / "comparacao_validacao_p10.csv"
)
CAMINHO_RESUMO = (
    PASTA_RESULTADOS / "resumo_validacao_p10.csv"
)


def exigir_colunas(dados, colunas, nome_arquivo):
    ausentes = sorted(set(colunas) - set(dados.columns))
    if ausentes:
        raise ValueError(
            f"Colunas ausentes em {nome_arquivo}: {ausentes}"
        )


def resumir_metodo(dados, metodo, coluna_estimativa):
    erros = (
        dados[coluna_estimativa]
        - dados["distancia_real_m"]
    )
    erros_absolutos = erros.abs()

    return {
        "metodo": metodo,
        "quantidade_medicoes": len(dados),
        "vies_m": erros.mean(),
        "mae_m": erros_absolutos.mean(),
        "rmse_m": np.sqrt(np.mean(np.square(erros))),
        "mediana_erro_absoluto_m": erros_absolutos.median(),
        "mape_percentual": (
            100
            * erros_absolutos
            / dados["distancia_real_m"]
        ).mean(),
        "erro_maximo_absoluto_m": erros_absolutos.max(),
    }


previsoes = pd.read_csv(CAMINHO_PREVISOES)
gabarito = pd.read_csv(CAMINHO_GABARITO)

exigir_colunas(
    previsoes,
    {
        "imagem",
        "deteccao",
        "classe",
        "distancia_centro_api_m",
        "distancia_obb_p10_m",
        "amplitude_p90_p10_m",
    },
    CAMINHO_PREVISOES.name,
)

exigir_colunas(
    gabarito,
    {
        "imagem",
        "numero_rosa",
        "posicao_relativa",
        "deteccao",
        "classe",
        "distancia_real_m",
    },
    CAMINHO_GABARITO.name,
)

if len(gabarito) != 6:
    raise ValueError(
        f"Esperadas 6 medições no gabarito; encontradas {len(gabarito)}."
    )

if gabarito[["imagem", "deteccao"]].duplicated().any():
    raise ValueError(
        "Há correspondências duplicadas no gabarito."
    )

if previsoes[["imagem", "deteccao"]].duplicated().any():
    raise ValueError(
        "Há previsões duplicadas para a mesma detecção."
    )

if (gabarito["distancia_real_m"] <= 0).any():
    raise ValueError(
        "Todas as distâncias reais devem ser positivas."
    )

dados = gabarito.merge(
    previsoes[
        [
            "imagem",
            "deteccao",
            "classe",
            "distancia_centro_api_m",
            "distancia_obb_p10_m",
            "amplitude_p90_p10_m",
        ]
    ],
    on=["imagem", "deteccao"],
    how="left",
    validate="one_to_one",
    indicator=True,
    suffixes=("_gabarito", "_previsao"),
)

faltantes = dados.loc[
    dados["_merge"] != "both",
    ["imagem", "deteccao"],
]

if not faltantes.empty:
    raise ValueError(
        "Detecções do gabarito sem previsão correspondente:\n"
        + faltantes.to_string(index=False)
    )

dados = dados.drop(columns="_merge")

classes_diferentes = dados.loc[
    dados["classe_gabarito"].str.strip()
    != dados["classe_previsao"].str.strip(),
    [
        "imagem",
        "deteccao",
        "classe_gabarito",
        "classe_previsao",
    ],
]

if not classes_diferentes.empty:
    raise ValueError(
        "Classes divergentes entre gabarito e previsões:\n"
        + classes_diferentes.to_string(index=False)
    )

dados = dados.sort_values(
    ["imagem", "numero_rosa"]
).reset_index(drop=True)

comparacao = dados[
    [
        "imagem",
        "numero_rosa",
        "posicao_relativa",
        "deteccao",
        "classe_previsao",
        "distancia_real_m",
        "distancia_centro_api_m",
        "distancia_obb_p10_m",
        "amplitude_p90_p10_m",
    ]
].rename(
    columns={"classe_previsao": "classe"}
)

for metodo, coluna_estimativa in {
    "centro_api": "distancia_centro_api_m",
    "obb_p10": "distancia_obb_p10_m",
}.items():
    erro = (
        comparacao[coluna_estimativa]
        - comparacao["distancia_real_m"]
    )

    comparacao[f"erro_assinado_{metodo}_m"] = erro
    comparacao[f"erro_absoluto_{metodo}_m"] = erro.abs()
    comparacao[
        f"erro_percentual_absoluto_{metodo}"
    ] = (
        100
        * erro.abs()
        / comparacao["distancia_real_m"]
    )

resumo = pd.DataFrame(
    [
        resumir_metodo(
            comparacao,
            "centro_api",
            "distancia_centro_api_m",
        ),
        resumir_metodo(
            comparacao,
            "obb_p10",
            "distancia_obb_p10_m",
        ),
    ]
)

PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

comparacao.to_csv(
    CAMINHO_COMPARACAO,
    index=False,
    float_format="%.6f",
)
resumo.to_csv(
    CAMINHO_RESUMO,
    index=False,
    float_format="%.6f",
)

resumo_por_metodo = resumo.set_index("metodo")
mae_centro = resumo_por_metodo.loc["centro_api", "mae_m"]
mae_p10 = resumo_por_metodo.loc["obb_p10", "mae_m"]
rmse_centro = resumo_por_metodo.loc["centro_api", "rmse_m"]
rmse_p10 = resumo_por_metodo.loc["obb_p10", "rmse_m"]

reducao_mae = 100 * (mae_centro - mae_p10) / mae_centro
reducao_rmse = 100 * (rmse_centro - rmse_p10) / rmse_centro

vitorias_p10 = int(
    (
        comparacao["erro_absoluto_obb_p10_m"]
        < comparacao["erro_absoluto_centro_api_m"]
    ).sum()
)

print("\nValidação independente por medição:\n")
print(
    comparacao[
        [
            "imagem",
            "numero_rosa",
            "deteccao",
            "distancia_real_m",
            "distancia_centro_api_m",
            "erro_absoluto_centro_api_m",
            "distancia_obb_p10_m",
            "erro_absoluto_obb_p10_m",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print("\nResumo da validação independente:\n")
print(resumo.round(4).to_string(index=False))

print(
    f"\nRedução do MAE com P10: {reducao_mae:.2f}%"
)
print(
    f"Redução do RMSE com P10: {reducao_rmse:.2f}%"
)
print(
    "Medições em que o P10 teve menor erro absoluto: "
    f"{vitorias_p10}/{len(comparacao)}"
)

print(f"\nComparação salva em: {CAMINHO_COMPARACAO}")
print(f"Resumo salvo em: {CAMINHO_RESUMO}")
