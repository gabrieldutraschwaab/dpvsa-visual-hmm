from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]

CAMINHO_MEDIDAS = (
    RAIZ
    / "dados"
    / "medidas_reais"
    / "distancias_controladas.csv"
)

CAMINHO_CORRESPONDENCIAS = (
    RAIZ
    / "dados"
    / "medidas_reais"
    / "correspondencias_deteccoes.csv"
)

CAMINHO_DETECCOES = (
    RAIZ
    / "dados"
    / "processados"
    / "deteccoes_imagens_controladas.csv"
)

CAMINHO_COMPARACAO = (
    RAIZ
    / "resultados"
    / "comparacao_distancias_controladas.csv"
)

CAMINHO_RESUMO = (
    RAIZ
    / "resultados"
    / "resumo_erros_distancias.csv"
)


for caminho in (
    CAMINHO_MEDIDAS,
    CAMINHO_CORRESPONDENCIAS,
    CAMINHO_DETECCOES,
):
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )


medidas = pd.read_csv(CAMINHO_MEDIDAS)
correspondencias = pd.read_csv(
    CAMINHO_CORRESPONDENCIAS
)
deteccoes = pd.read_csv(CAMINHO_DETECCOES)


chave_medicao = ["imagem", "posicao_relativa"]

if medidas.duplicated(chave_medicao).any():
    raise ValueError(
        "Existem medidas reais duplicadas."
    )

if correspondencias.duplicated(chave_medicao).any():
    raise ValueError(
        "Existem correspondências duplicadas."
    )

if deteccoes.duplicated(
    ["imagem", "deteccao"]
).any():
    raise ValueError(
        "Existem identificadores de detecção duplicados."
    )


comparacao = medidas.merge(
    correspondencias,
    on=chave_medicao,
    how="inner",
    validate="one_to_one",
)

comparacao = comparacao.merge(
    deteccoes[
        [
            "imagem",
            "deteccao",
            "classe",
            "camera_z",
            "distancia_camera",
        ]
    ],
    on=["imagem", "deteccao"],
    how="left",
    validate="many_to_one",
    suffixes=("_marcada", "_api"),
)


if comparacao["distancia_camera"].isna().any():
    raise ValueError(
        "Alguma correspondência não encontrou uma detecção."
    )

classes_divergentes = (
    comparacao["classe_marcada"]
    != comparacao["classe_api"]
)

if classes_divergentes.any():
    raise ValueError(
        "A classe registrada não confere com a API."
    )


comparacao = comparacao.rename(
    columns={
        "classe_api": "classe",
        "distancia_camera": "distancia_estimada_m",
    }
)

comparacao["erro_assinado_m"] = (
    comparacao["distancia_estimada_m"]
    - comparacao["distancia_real_m"]
)

comparacao["erro_absoluto_m"] = (
    comparacao["erro_assinado_m"].abs()
)

comparacao["erro_assinado_cm"] = (
    comparacao["erro_assinado_m"] * 100
)

comparacao["erro_absoluto_cm"] = (
    comparacao["erro_absoluto_m"] * 100
)

comparacao["erro_percentual_assinado"] = (
    comparacao["erro_assinado_m"]
    / comparacao["distancia_real_m"]
    * 100
)

comparacao["erro_percentual_absoluto"] = (
    comparacao["erro_absoluto_m"]
    / comparacao["distancia_real_m"]
    * 100
)


comparacao = comparacao[
    [
        "imagem",
        "posicao_relativa",
        "numero_rosa",
        "deteccao",
        "classe",
        "criterio_associacao",
        "referencia_medicao",
        "distancia_real_m",
        "camera_z",
        "distancia_estimada_m",
        "erro_assinado_m",
        "erro_absoluto_m",
        "erro_assinado_cm",
        "erro_absoluto_cm",
        "erro_percentual_assinado",
        "erro_percentual_absoluto",
    ]
].sort_values(
    ["imagem", "numero_rosa"]
)


resumo = pd.DataFrame(
    [
        {
            "quantidade_medicoes": len(comparacao),
            "vies_m": comparacao[
                "erro_assinado_m"
            ].mean(),
            "mae_m": comparacao[
                "erro_absoluto_m"
            ].mean(),
            "rmse_m": np.sqrt(
                np.mean(
                    comparacao[
                        "erro_assinado_m"
                    ] ** 2
                )
            ),
            "mediana_erro_absoluto_m": comparacao[
                "erro_absoluto_m"
            ].median(),
            "mape_percentual": comparacao[
                "erro_percentual_absoluto"
            ].mean(),
        }
    ]
)


CAMINHO_COMPARACAO.parent.mkdir(
    parents=True,
    exist_ok=True,
)

comparacao.to_csv(
    CAMINHO_COMPARACAO,
    index=False,
)

resumo.to_csv(
    CAMINHO_RESUMO,
    index=False,
)


print("Comparação por válvula:")
print()
print(
    comparacao[
        [
            "imagem",
            "numero_rosa",
            "distancia_real_m",
            "distancia_estimada_m",
            "erro_assinado_cm",
            "erro_percentual_absoluto",
        ]
    ].round(4).to_string(index=False)
)

print()
print("Resumo descritivo:")
print()
print(
    resumo.round(4).to_string(index=False)
)

print()
print("Arquivo de comparação:", CAMINHO_COMPARACAO)
print("Arquivo de resumo:", CAMINHO_RESUMO)
