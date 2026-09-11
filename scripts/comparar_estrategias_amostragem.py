from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
RAIZ_API = RAIZ.parent / "api-digital-twin"

sys.path.insert(0, str(RAIZ_API))

from app import main as api  # noqa: E402


FX = 1268.9369
FY = 1272.6680
CX = 457.3792
CY = 841.9177

ORDEM_METODOS = [
    "centro_api",
    "obb_p10",
    "obb_p25",
    "obb_mediana",
]

CAMINHO_CORRESPONDENCIAS = (
    RAIZ
    / "dados"
    / "medidas_reais"
    / "correspondencias_deteccoes.csv"
)

CAMINHO_MEDIDAS = (
    RAIZ
    / "dados"
    / "medidas_reais"
    / "distancias_controladas.csv"
)

CAMINHO_REFERENCIA_API = (
    RAIZ
    / "dados"
    / "processados"
    / "deteccoes_imagens_controladas.csv"
)

SAIDA_COMPARACAO = (
    RAIZ
    / "resultados"
    / "comparacao_estrategias_amostragem.csv"
)

SAIDA_RESUMO = (
    RAIZ
    / "resultados"
    / "resumo_estrategias_amostragem.csv"
)

SAIDA_HETEROGENEIDADE = (
    RAIZ
    / "resultados"
    / "heterogeneidade_profundidade_obb.csv"
)


def ajustar_z(valor):
    return max(
        0.05,
        api.DEPTH_SCALE * float(valor)
        + api.DEPTH_BIAS_METERS,
    )


def valores_validos(valores):
    valores = np.asarray(valores, dtype=np.float32)

    return valores[
        np.isfinite(valores)
        & (valores > 0.05)
        & (valores <= 20.0)
    ]


correspondencias = pd.read_csv(
    CAMINHO_CORRESPONDENCIAS
)

medidas = pd.read_csv(
    CAMINHO_MEDIDAS
)

referencia_api = pd.read_csv(
    CAMINHO_REFERENCIA_API
)

alvos = correspondencias.merge(
    medidas,
    on=["imagem", "posicao_relativa"],
    how="inner",
    validate="one_to_one",
)

if len(alvos) != 4:
    raise RuntimeError(
        f"Eram esperados 4 alvos, mas foram encontrados {len(alvos)}."
    )

comparacoes = []
heterogeneidades = []

for nome_imagem, alvos_imagem in alvos.groupby(
    "imagem",
    sort=True,
):
    caminho_imagem = (
        RAIZ_API
        / "test_images"
        / nome_imagem
    )

    frame = cv2.imread(str(caminho_imagem))

    if frame is None:
        raise FileNotFoundError(
            f"Não foi possível abrir: {caminho_imagem}"
        )

    print(f"Processando {nome_imagem}...")

    deteccoes = api.rodar_yolo_obb(frame)
    mapa = api.rodar_depth(frame)

    for alvo in alvos_imagem.itertuples(index=False):
        numero_deteccao = int(alvo.deteccao)
        indice = numero_deteccao - 1

        if indice < 0 or indice >= len(deteccoes):
            raise RuntimeError(
                f"Detecção {numero_deteccao} inexistente "
                f"em {nome_imagem}."
            )

        deteccao = deteccoes[indice]

        if deteccao["classe"] != alvo.classe:
            raise RuntimeError(
                f"A ordem das detecções mudou em {nome_imagem}. "
                f"Esperado: {alvo.classe}. "
                f"Obtido: {deteccao['classe']}."
            )

        centro_x, centro_y = deteccao["centro"]
        largura, altura = deteccao["dimensoes"]

        poligono = np.asarray(
            deteccao["pontos_poligono"],
            dtype=np.int32,
        ).reshape(-1, 2)

        mascara = np.zeros(
            mapa.shape[:2],
            dtype=np.uint8,
        )

        cv2.fillPoly(
            mascara,
            [poligono],
            255,
        )

        valores_obb = valores_validos(
            mapa[mascara.astype(bool)]
        )

        if valores_obb.size == 0:
            raise RuntimeError(
                f"OBB sem profundidades válidas em "
                f"{nome_imagem}, detecção {numero_deteccao}."
            )

        p10, p25, mediana, p75, p90 = np.percentile(
            valores_obb,
            [10, 25, 50, 75, 90],
        )

        z_centro_api = ajustar_z(
            api.extrair_profundidade_metrica(
                mapa,
                centro_x,
                centro_y,
                largura,
                altura,
            )
        )

        fator_raio = np.sqrt(
            1.0
            + ((centro_x - CX) / FX) ** 2
            + ((centro_y - CY) / FY) ** 2
        )

        metodos = {
            "centro_api": z_centro_api,
            "obb_p10": ajustar_z(p10),
            "obb_p25": ajustar_z(p25),
            "obb_mediana": ajustar_z(mediana),
        }

        distancia_centro_api = (
            metodos["centro_api"] * fator_raio
        )

        referencia = referencia_api[
            (referencia_api["imagem"] == nome_imagem)
            & (
                referencia_api["deteccao"]
                == numero_deteccao
            )
        ]

        if len(referencia) != 1:
            raise RuntimeError(
                f"Referência anterior ausente para "
                f"{nome_imagem}, detecção {numero_deteccao}."
            )

        distancia_anterior = float(
            referencia.iloc[0]["distancia_camera"]
        )

        if (
            abs(
                distancia_centro_api
                - distancia_anterior
            )
            > 0.01
        ):
            raise RuntimeError(
                f"O resultado atual de {nome_imagem}, "
                f"detecção {numero_deteccao}, não reproduziu "
                f"a execução anterior."
            )

        distancia_real = float(
            alvo.distancia_real_m
        )

        for metodo, z_estimado in metodos.items():
            distancia_estimada = (
                z_estimado * fator_raio
            )

            erro_assinado = (
                distancia_estimada
                - distancia_real
            )

            comparacoes.append(
                {
                    "imagem": nome_imagem,
                    "numero_rosa": int(
                        alvo.numero_rosa
                    ),
                    "posicao_relativa": (
                        alvo.posicao_relativa
                    ),
                    "deteccao": numero_deteccao,
                    "classe": deteccao["classe"],
                    "metodo": metodo,
                    "z_estimado_m": z_estimado,
                    "distancia_real_m": (
                        distancia_real
                    ),
                    "distancia_estimada_m": (
                        distancia_estimada
                    ),
                    "erro_assinado_m": (
                        erro_assinado
                    ),
                    "erro_absoluto_m": abs(
                        erro_assinado
                    ),
                    "erro_percentual_absoluto": (
                        abs(erro_assinado)
                        / distancia_real
                        * 100.0
                    ),
                }
            )

        heterogeneidades.append(
            {
                "imagem": nome_imagem,
                "numero_rosa": int(
                    alvo.numero_rosa
                ),
                "deteccao": numero_deteccao,
                "classe": deteccao["classe"],
                "amostras_obb": int(
                    valores_obb.size
                ),
                "z_p10_m": ajustar_z(p10),
                "z_p25_m": ajustar_z(p25),
                "z_mediana_m": ajustar_z(
                    mediana
                ),
                "z_p75_m": ajustar_z(p75),
                "z_p90_m": ajustar_z(p90),
                "amplitude_p90_p10_m": (
                    ajustar_z(p90)
                    - ajustar_z(p10)
                ),
                "iqr_m": (
                    ajustar_z(p75)
                    - ajustar_z(p25)
                ),
                "amplitude_relativa": (
                    (
                        ajustar_z(p90)
                        - ajustar_z(p10)
                    )
                    / ajustar_z(mediana)
                ),
            }
        )


comparacao = pd.DataFrame(comparacoes).sort_values(
    ["imagem", "numero_rosa", "metodo"]
)

heterogeneidade = pd.DataFrame(
    heterogeneidades
).sort_values(
    ["imagem", "numero_rosa"]
)

linhas_resumo = []

for metodo in ORDEM_METODOS:
    grupo = comparacao[
        comparacao["metodo"] == metodo
    ]

    erros = grupo["erro_assinado_m"].to_numpy()
    erros_absolutos = np.abs(erros)

    linhas_resumo.append(
        {
            "metodo": metodo,
            "quantidade_medicoes": len(grupo),
            "vies_m": float(np.mean(erros)),
            "mae_m": float(
                np.mean(erros_absolutos)
            ),
            "rmse_m": float(
                np.sqrt(np.mean(erros**2))
            ),
            "mediana_erro_absoluto_m": float(
                np.median(erros_absolutos)
            ),
            "mape_percentual": float(
                np.mean(
                    grupo[
                        "erro_percentual_absoluto"
                    ]
                )
            ),
        }
    )

resumo = pd.DataFrame(linhas_resumo)

SAIDA_COMPARACAO.parent.mkdir(
    parents=True,
    exist_ok=True,
)

comparacao.to_csv(
    SAIDA_COMPARACAO,
    index=False,
)

resumo.to_csv(
    SAIDA_RESUMO,
    index=False,
)

heterogeneidade.to_csv(
    SAIDA_HETEROGENEIDADE,
    index=False,
)

tabela_estimativas = comparacao.pivot(
    index=[
        "imagem",
        "numero_rosa",
        "distancia_real_m",
    ],
    columns="metodo",
    values="distancia_estimada_m",
).reset_index()

tabela_estimativas.columns.name = None

tabela_estimativas = tabela_estimativas[
    [
        "imagem",
        "numero_rosa",
        "distancia_real_m",
        *ORDEM_METODOS,
    ]
]

print()
print("Estimativas por estratégia:")
print(
    tabela_estimativas
    .round(4)
    .to_string(index=False)
)

print()
print("Resumo exploratório:")
print(
    resumo
    .round(4)
    .to_string(index=False)
)

print()
print("Heterogeneidade dentro das OBBs:")
print(
    heterogeneidade[
        [
            "imagem",
            "numero_rosa",
            "z_p10_m",
            "z_mediana_m",
            "z_p90_m",
            "amplitude_p90_p10_m",
            "iqr_m",
            "amplitude_relativa",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print()
print(
    "ATENÇÃO: estas quatro medições foram usadas "
    "para explorar os métodos. Elas não constituem "
    "uma validação independente."
)

print()
print("Comparação salva em:", SAIDA_COMPARACAO)
print("Resumo salvo em:", SAIDA_RESUMO)
print(
    "Heterogeneidade salva em:",
    SAIDA_HETEROGENEIDADE,
)
