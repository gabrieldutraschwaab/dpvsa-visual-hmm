from pathlib import Path
import hashlib
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
RAIZ_API = RAIZ.parent / "api-digital-twin"

sys.path.insert(0, str(RAIZ_API))

from app import main as api  # noqa: E402


PASTA_IMAGENS = (
    RAIZ
    / "dados"
    / "imagens_valvulas"
    / "validacao_independente"
)

SAIDA_PREVISOES = (
    RAIZ
    / "dados"
    / "processados"
    / "previsoes_validacao_p10.csv"
)

SAIDA_MANIFESTO = (
    RAIZ
    / "dados"
    / "processados"
    / "manifesto_validacao_independente.csv"
)

PASTA_RESULTADOS = (
    RAIZ
    / "resultados"
    / "validacao_independente"
)

FX = 1268.9369
FY = 1272.6680
CX = 457.3792
CY = 841.9177

LARGURA_ESPERADA = 900
ALTURA_ESPERADA = 1600

EXTENSOES = {
    ".jpg",
    ".jpeg",
    ".png",
}


def calcular_sha256(caminho):
    return hashlib.sha256(
        caminho.read_bytes()
    ).hexdigest()


def ajustar_z(valor):
    return max(
        0.05,
        api.DEPTH_SCALE * float(valor)
        + api.DEPTH_BIAS_METERS,
    )


def valores_validos(valores):
    valores = np.asarray(
        valores,
        dtype=np.float32,
    )

    return valores[
        np.isfinite(valores)
        & (valores > 0.05)
        & (valores <= 20.0)
    ]


def desenhar_identificacao(
    imagem,
    poligono,
    centro_x,
    centro_y,
    numero,
    classe,
):
    altura_imagem, largura_imagem = (
        imagem.shape[:2]
    )

    cv2.polylines(
        imagem,
        [poligono],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )

    cx = int(
        np.clip(
            round(centro_x),
            0,
            largura_imagem - 1,
        )
    )

    cy = int(
        np.clip(
            round(centro_y),
            0,
            altura_imagem - 1,
        )
    )

    cv2.circle(
        imagem,
        (cx, cy),
        7,
        (255, 0, 255),
        -1,
    )

    texto = f"{numero}: {classe}"
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.65
    espessura = 2

    (largura_texto, altura_texto), _ = (
        cv2.getTextSize(
            texto,
            fonte,
            escala,
            espessura,
        )
    )

    x = int(np.min(poligono[:, 0]))
    y = int(np.min(poligono[:, 1])) - 8

    x = int(
        np.clip(
            x,
            4,
            max(
                4,
                largura_imagem
                - largura_texto
                - 8,
            ),
        )
    )

    y = int(
        np.clip(
            y,
            altura_texto + 8,
            altura_imagem - 5,
        )
    )

    cv2.putText(
        imagem,
        texto,
        (x, y),
        fonte,
        escala,
        (0, 0, 0),
        espessura + 3,
        cv2.LINE_AA,
    )

    cv2.putText(
        imagem,
        texto,
        (x, y),
        fonte,
        escala,
        (255, 255, 255),
        espessura,
        cv2.LINE_AA,
    )


imagens = sorted(
    caminho
    for caminho in PASTA_IMAGENS.iterdir()
    if (
        caminho.is_file()
        and caminho.suffix.lower()
        in EXTENSOES
    )
)

if not imagens:
    raise RuntimeError(
        f"Nenhuma imagem encontrada em: {PASTA_IMAGENS}"
    )

commit_api = subprocess.run(
    [
        "git",
        "-C",
        str(RAIZ_API),
        "rev-parse",
        "HEAD",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()

SAIDA_PREVISOES.parent.mkdir(
    parents=True,
    exist_ok=True,
)

PASTA_RESULTADOS.mkdir(
    parents=True,
    exist_ok=True,
)

previsoes = []
manifesto = []

for caminho_imagem in imagens:
    frame = cv2.imread(
        str(caminho_imagem)
    )

    if frame is None:
        raise RuntimeError(
            f"Não foi possível abrir: {caminho_imagem}"
        )

    altura, largura = frame.shape[:2]

    if (
        largura != LARGURA_ESPERADA
        or altura != ALTURA_ESPERADA
    ):
        raise RuntimeError(
            f"{caminho_imagem.name} possui "
            f"{largura}x{altura}, mas eram esperados "
            f"{LARGURA_ESPERADA}x{ALTURA_ESPERADA}. "
            "Os intrínsecos não podem ser aplicados "
            "diretamente."
        )

    print(
        f"Processando {caminho_imagem.name}..."
    )

    deteccoes = api.rodar_yolo_obb(frame)
    mapa = api.rodar_depth(frame)

    imagem_numerada = frame.copy()

    manifesto.append(
        {
            "imagem": caminho_imagem.name,
            "sha256": calcular_sha256(
                caminho_imagem
            ),
            "tamanho_bytes": (
                caminho_imagem.stat().st_size
            ),
            "largura_px": largura,
            "altura_px": altura,
            "quantidade_deteccoes": len(
                deteccoes
            ),
            "origem_arquivo": "WhatsApp",
            "metodo_congelado": "obb_p10",
            "quantil": 0.10,
            "fx": FX,
            "fy": FY,
            "cx": CX,
            "cy": CY,
            "depth_scale": api.DEPTH_SCALE,
            "depth_bias_meters": (
                api.DEPTH_BIAS_METERS
            ),
            "modelo_profundidade": (
                api.DEPTH_MODEL_ID
            ),
            "pesos_yolo": (
                api.YOLO_WEIGHTS.name
            ),
            "commit_api": commit_api,
        }
    )

    for numero, deteccao in enumerate(
        deteccoes,
        start=1,
    ):
        centro_x, centro_y = (
            deteccao["centro"]
        )

        largura_obb, altura_obb = (
            deteccao["dimensoes"]
        )

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
                f"Sem profundidades válidas em "
                f"{caminho_imagem.name}, "
                f"detecção {numero}."
            )

        p10, p90 = np.percentile(
            valores_obb,
            [10, 90],
        )

        z_p10 = ajustar_z(p10)

        z_centro_api = ajustar_z(
            api.extrair_profundidade_metrica(
                mapa,
                centro_x,
                centro_y,
                largura_obb,
                altura_obb,
            )
        )

        x_p10 = (
            (centro_x - CX)
            * (z_p10 / FX)
        )

        y_p10 = -(
            (centro_y - CY)
            * (z_p10 / FY)
        )

        distancia_p10 = float(
            np.linalg.norm(
                [x_p10, y_p10, z_p10]
            )
        )

        x_centro_api = (
            (centro_x - CX)
            * (z_centro_api / FX)
        )

        y_centro_api = -(
            (centro_y - CY)
            * (z_centro_api / FY)
        )

        distancia_centro_api = float(
            np.linalg.norm(
                [
                    x_centro_api,
                    y_centro_api,
                    z_centro_api,
                ]
            )
        )

        previsoes.append(
            {
                "imagem": (
                    caminho_imagem.name
                ),
                "deteccao": numero,
                "classe": deteccao["classe"],
                "centro_x_px": centro_x,
                "centro_y_px": centro_y,
                "largura_obb_px": largura_obb,
                "altura_obb_px": altura_obb,
                "theta_rad": (
                    deteccao["theta"]
                ),
                "z_centro_api_m": (
                    z_centro_api
                ),
                "distancia_centro_api_m": (
                    distancia_centro_api
                ),
                "z_obb_p10_m": z_p10,
                "camera_x_p10_m": x_p10,
                "camera_y_p10_m": y_p10,
                "camera_z_p10_m": z_p10,
                "distancia_obb_p10_m": (
                    distancia_p10
                ),
                "amplitude_p90_p10_m": (
                    ajustar_z(p90)
                    - ajustar_z(p10)
                ),
            }
        )

        desenhar_identificacao(
            imagem_numerada,
            poligono,
            centro_x,
            centro_y,
            numero,
            deteccao["classe"],
        )

    caminho_saida_imagem = (
        PASTA_RESULTADOS
        / (
            caminho_imagem.stem
            + "_deteccoes_numeradas.jpg"
        )
    )

    sucesso = cv2.imwrite(
        str(caminho_saida_imagem),
        imagem_numerada,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not sucesso:
        raise RuntimeError(
            f"Não foi possível salvar: "
            f"{caminho_saida_imagem}"
        )

    print(
        f"  Detecções: {len(deteccoes)}"
    )

    print(
        f"  Imagem numerada: "
        f"{caminho_saida_imagem}"
    )


tabela_previsoes = pd.DataFrame(
    previsoes
)

tabela_manifesto = pd.DataFrame(
    manifesto
)

tabela_previsoes.to_csv(
    SAIDA_PREVISOES,
    index=False,
)

tabela_manifesto.to_csv(
    SAIDA_MANIFESTO,
    index=False,
)

print()
print(
    "Previsões P10 congeladas e salvas, "
    "sem utilização das distâncias reais."
)

print(
    "Arquivo de previsões:",
    SAIDA_PREVISOES,
)

print(
    "Manifesto:",
    SAIDA_MANIFESTO,
)

print()
print(
    "Não abra a tabela de previsões antes "
    "de registrar a correspondência visual "
    "das válvulas medidas."
)
