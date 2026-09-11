from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
RAIZ_API = RAIZ.parent / "api-digital-twin"

if not RAIZ_API.is_dir():
    raise FileNotFoundError(
        f"Repositório da API não encontrado em: {RAIZ_API}"
    )

sys.path.insert(0, str(RAIZ_API))

from app import main as api  # noqa: E402


FX = 1268.9369
FY = 1272.6680
CX = 457.3792
CY = 841.9177

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

CAMINHO_DETECCOES_ANTERIORES = (
    RAIZ
    / "dados"
    / "processados"
    / "deteccoes_imagens_controladas.csv"
)

SAIDA = (
    RAIZ
    / "resultados"
    / "diagnostico_amostragem_profundidade.csv"
)


def valores_validos(valores):
    valores = np.asarray(valores, dtype=np.float32)

    return valores[
        np.isfinite(valores)
        & (valores > 0.05)
        & (valores <= 20.0)
    ]


def aplicar_ajuste(z):
    return max(
        0.05,
        api.DEPTH_SCALE * float(z) + api.DEPTH_BIAS_METERS,
    )


def filtrar_por_mad(valores):
    valores = valores.copy()
    mediana = float(np.median(valores))
    mad = float(np.median(np.abs(valores - mediana)))

    if mad > 1e-6:
        limite = 3.0 * 1.4826 * mad
        filtrados = valores[
            np.abs(valores - mediana) <= limite
        ]

        if filtrados.size:
            valores = filtrados

    return valores


correspondencias = pd.read_csv(CAMINHO_CORRESPONDENCIAS)
medidas = pd.read_csv(CAMINHO_MEDIDAS)
deteccoes_anteriores = pd.read_csv(
    CAMINHO_DETECCOES_ANTERIORES
)

alvos = correspondencias.merge(
    medidas,
    on=["imagem", "posicao_relativa"],
    how="inner",
    validate="one_to_one",
)

if len(alvos) != 4:
    raise RuntimeError(
        f"Eram esperadas 4 correspondências, mas foram encontradas {len(alvos)}."
    )

linhas = []

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

    print(f"Processando mapa numérico de {nome_imagem}...")

    deteccoes = api.rodar_yolo_obb(frame)
    mapa_profundidade = api.rodar_depth(frame)

    for alvo in alvos_imagem.itertuples(index=False):
        numero_deteccao = int(alvo.deteccao)
        indice = numero_deteccao - 1

        if indice < 0 or indice >= len(deteccoes):
            raise RuntimeError(
                f"Detecção {numero_deteccao} inexistente em {nome_imagem}."
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

        altura_mapa, largura_mapa = mapa_profundidade.shape[:2]

        cx_pixel = int(
            np.clip(round(centro_x), 0, largura_mapa - 1)
        )
        cy_pixel = int(
            np.clip(round(centro_y), 0, altura_mapa - 1)
        )

        raio = max(
            3,
            int(round(min(largura, altura) * 0.20)),
        )
        raio = min(raio, 25)

        x1 = max(0, cx_pixel - raio)
        x2 = min(largura_mapa, cx_pixel + raio + 1)
        y1 = max(0, cy_pixel - raio)
        y2 = min(altura_mapa, cy_pixel + raio + 1)

        recorte = mapa_profundidade[y1:y2, x1:x2]

        yy, xx = np.ogrid[y1:y2, x1:x2]
        mascara = (
            (xx - cx_pixel) ** 2
            + (yy - cy_pixel) ** 2
            <= raio**2
        )

        valores = valores_validos(recorte[mascara])

        if valores.size == 0:
            raise RuntimeError(
                f"Região sem profundidades válidas em "
                f"{nome_imagem}, detecção {numero_deteccao}."
            )

        valores_apos_mad = filtrar_por_mad(valores)

        percentis = np.percentile(
            valores,
            [0, 10, 25, 50, 75, 90, 100],
        )

        z_centro_bruto = float(
            mapa_profundidade[cy_pixel, cx_pixel]
        )

        if (
            np.isfinite(z_centro_bruto)
            and 0.05 < z_centro_bruto <= 20.0
        ):
            z_centro = aplicar_ajuste(z_centro_bruto)
        else:
            z_centro = np.nan

        z_api = aplicar_ajuste(
            float(np.median(valores_apos_mad))
        )

        z_api_funcao = aplicar_ajuste(
            api.extrair_profundidade_metrica(
                mapa_profundidade,
                centro_x,
                centro_y,
                largura,
                altura,
            )
        )

        if abs(z_api - z_api_funcao) > 1e-5:
            raise RuntimeError(
                "O diagnóstico não reproduziu o método atual da API."
            )

        fator_raio = np.sqrt(
            1.0
            + ((centro_x - CX) / FX) ** 2
            + ((centro_y - CY) / FY) ** 2
        )

        distancia_api = z_api * fator_raio

        if np.isfinite(z_centro):
            distancia_centro = z_centro * fator_raio
        else:
            distancia_centro = np.nan

        referencia = deteccoes_anteriores[
            (deteccoes_anteriores["imagem"] == nome_imagem)
            & (
                deteccoes_anteriores["deteccao"]
                == numero_deteccao
            )
        ]

        if len(referencia) != 1:
            raise RuntimeError(
                f"Referência anterior não encontrada para "
                f"{nome_imagem}, detecção {numero_deteccao}."
            )

        distancia_anterior = float(
            referencia.iloc[0]["distancia_camera"]
        )

        if abs(distancia_api - distancia_anterior) > 0.01:
            raise RuntimeError(
                f"A detecção de {nome_imagem} mudou de ordem ou valor. "
                f"Anterior: {distancia_anterior:.4f} m. "
                f"Atual: {distancia_api:.4f} m."
            )

        distancia_real = float(alvo.distancia_real_m)

        linhas.append(
            {
                "imagem": nome_imagem,
                "numero_rosa": int(alvo.numero_rosa),
                "posicao_relativa": alvo.posicao_relativa,
                "deteccao": numero_deteccao,
                "classe": deteccao["classe"],
                "centro_x_px": centro_x,
                "centro_y_px": centro_y,
                "raio_px": raio,
                "amostras_circulo": int(valores.size),
                "amostras_apos_mad": int(
                    valores_apos_mad.size
                ),
                "z_centro_m": z_centro,
                "z_min_m": aplicar_ajuste(percentis[0]),
                "z_p10_m": aplicar_ajuste(percentis[1]),
                "z_p25_m": aplicar_ajuste(percentis[2]),
                "z_mediana_bruta_m": aplicar_ajuste(
                    percentis[3]
                ),
                "z_p75_m": aplicar_ajuste(percentis[4]),
                "z_p90_m": aplicar_ajuste(percentis[5]),
                "z_max_m": aplicar_ajuste(percentis[6]),
                "z_api_m": z_api,
                "distancia_centro_m": distancia_centro,
                "distancia_api_m": distancia_api,
                "distancia_real_m": distancia_real,
                "erro_api_cm": (
                    distancia_api - distancia_real
                ) * 100.0,
                "dispersao_p90_p10_m": (
                    aplicar_ajuste(percentis[5])
                    - aplicar_ajuste(percentis[1])
                ),
            }
        )


resultado = pd.DataFrame(linhas).sort_values(
    ["imagem", "numero_rosa"]
)

SAIDA.parent.mkdir(parents=True, exist_ok=True)
resultado.to_csv(SAIDA, index=False)

colunas_exibidas = [
    "imagem",
    "numero_rosa",
    "deteccao",
    "raio_px",
    "z_centro_m",
    "z_p10_m",
    "z_p25_m",
    "z_mediana_bruta_m",
    "z_p75_m",
    "z_p90_m",
    "z_api_m",
    "distancia_real_m",
    "distancia_api_m",
    "erro_api_cm",
]

print()
print("Diagnóstico da região usada pela API:")
print(
    resultado[colunas_exibidas]
    .round(4)
    .to_string(index=False)
)
print()
print("Arquivo salvo em:", SAIDA)
