from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
RAIZ_API = RAIZ.parent / "api-digital-twin"

sys.path.insert(0, str(RAIZ_API))

from app import main as api  # noqa: E402


IMAGEM = RAIZ_API / "test_images" / "img1.jpeg"
NUMERO_DETECCAO = 4
DISTANCIA_REAL_M = 0.83

FX = 1268.9369
FY = 1272.6680
CX = 457.3792
CY = 841.9177

SAIDA_CSV = (
    RAIZ
    / "resultados"
    / "diagnostico_perfil_obb_img1_alvo1.csv"
)

SAIDA_IMAGEM = (
    RAIZ
    / "resultados"
    / "img1_alvo1_perfil_obb.jpg"
)


def ajustar_z(valor):
    return max(
        0.05,
        api.DEPTH_SCALE * float(valor)
        + api.DEPTH_BIAS_METERS,
    )


def manter_validos(valores):
    valores = np.asarray(valores, dtype=np.float32)

    return valores[
        np.isfinite(valores)
        & (valores > 0.05)
        & (valores <= 20.0)
    ]


def amostrar_circulo(mapa, x, y, raio):
    altura_mapa, largura_mapa = mapa.shape[:2]

    cx = int(np.clip(round(x), 0, largura_mapa - 1))
    cy = int(np.clip(round(y), 0, altura_mapa - 1))

    x1 = max(0, cx - raio)
    x2 = min(largura_mapa, cx + raio + 1)
    y1 = max(0, cy - raio)
    y2 = min(altura_mapa, cy + raio + 1)

    recorte = mapa[y1:y2, x1:x2]

    yy, xx = np.ogrid[y1:y2, x1:x2]
    mascara = (
        (xx - cx) ** 2
        + (yy - cy) ** 2
        <= raio**2
    )

    valores = manter_validos(recorte[mascara])

    if valores.size == 0:
        raise RuntimeError(
            f"Sem valores válidos ao redor de ({cx}, {cy})."
        )

    p10, mediana, p90 = np.percentile(
        valores,
        [10, 50, 90],
    )

    return {
        "x_px": cx,
        "y_px": cy,
        "amostras": int(valores.size),
        "z_p10_m": ajustar_z(p10),
        "z_mediana_m": ajustar_z(mediana),
        "z_p90_m": ajustar_z(p90),
    }


frame = cv2.imread(str(IMAGEM))

if frame is None:
    raise FileNotFoundError(
        f"Não foi possível abrir: {IMAGEM}"
    )

deteccoes = api.rodar_yolo_obb(frame)

if len(deteccoes) < NUMERO_DETECCAO:
    raise RuntimeError(
        f"A detecção {NUMERO_DETECCAO} não foi encontrada."
    )

deteccao = deteccoes[NUMERO_DETECCAO - 1]

if deteccao["classe"] != "Sphere Valve S":
    raise RuntimeError(
        "A ordem das detecções mudou. "
        f"Classe obtida: {deteccao['classe']}"
    )

mapa = api.rodar_depth(frame)

centro_x, centro_y = deteccao["centro"]
largura, altura = deteccao["dimensoes"]
theta = deteccao["theta"]

if largura >= altura:
    comprimento = largura
    lado_curto = altura
    direcao_x = np.cos(theta)
    direcao_y = np.sin(theta)
else:
    comprimento = altura
    lado_curto = largura
    direcao_x = -np.sin(theta)
    direcao_y = np.cos(theta)

raio_amostra = max(
    2,
    min(5, int(round(lado_curto * 0.08))),
)

fator_centro = np.sqrt(
    1.0
    + ((centro_x - CX) / FX) ** 2
    + ((centro_y - CY) / FY) ** 2
)

z_real_esperado = DISTANCIA_REAL_M / fator_centro

fracoes = np.linspace(-0.40, 0.40, 9)
linhas = []

imagem_anotada = frame.copy()

poligono = np.asarray(
    deteccao["pontos_poligono"],
    dtype=np.int32,
).reshape(-1, 2)

cv2.polylines(
    imagem_anotada,
    [poligono],
    isClosed=True,
    color=(0, 255, 0),
    thickness=3,
)

for numero, fracao in enumerate(fracoes, start=1):
    x = centro_x + fracao * comprimento * direcao_x
    y = centro_y + fracao * comprimento * direcao_y

    amostra = amostrar_circulo(
        mapa,
        x,
        y,
        raio_amostra,
    )

    fator_raio = np.sqrt(
        1.0
        + ((x - CX) / FX) ** 2
        + ((y - CY) / FY) ** 2
    )

    distancia_mediana = (
        amostra["z_mediana_m"] * fator_raio
    )

    linhas.append(
        {
            "ponto": numero,
            "fracao_eixo_maior": fracao,
            **amostra,
            "distancia_mediana_m": distancia_mediana,
        }
    )

    ponto = (amostra["x_px"], amostra["y_px"])

    cv2.circle(
        imagem_anotada,
        ponto,
        raio_amostra + 2,
        (255, 0, 255),
        2,
    )

    cv2.putText(
        imagem_anotada,
        str(numero),
        (ponto[0] + 6, ponto[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


mascara_obb = np.zeros(
    mapa.shape[:2],
    dtype=np.uint8,
)

cv2.fillPoly(
    mascara_obb,
    [poligono],
    255,
)

valores_obb = manter_validos(
    mapa[mascara_obb.astype(bool)]
)

if valores_obb.size == 0:
    raise RuntimeError(
        "A OBB não contém profundidades válidas."
    )

percentis_obb = np.percentile(
    valores_obb,
    [0, 10, 25, 50, 75, 90, 100],
)

resultado = pd.DataFrame(linhas)

SAIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
resultado.to_csv(SAIDA_CSV, index=False)

if not cv2.imwrite(str(SAIDA_IMAGEM), imagem_anotada):
    raise RuntimeError(
        f"Não foi possível salvar: {SAIDA_IMAGEM}"
    )

print()
print("Alvo: img1.jpeg, número rosa 1, detecção 4")
print(f"Classe: {deteccao['classe']}")
print(f"Centro da OBB: ({centro_x:.1f}, {centro_y:.1f})")
print(f"Dimensões da OBB: {largura:.1f} x {altura:.1f} px")
print(f"Raio de cada amostra: {raio_amostra} px")
print(
    "Z esperado no centro a partir da distância real: "
    f"{z_real_esperado:.4f} m"
)

print()
print("Distribuição em toda a OBB:")
print(f"mínimo:  {ajustar_z(percentis_obb[0]):.4f} m")
print(f"P10:     {ajustar_z(percentis_obb[1]):.4f} m")
print(f"P25:     {ajustar_z(percentis_obb[2]):.4f} m")
print(f"mediana: {ajustar_z(percentis_obb[3]):.4f} m")
print(f"P75:     {ajustar_z(percentis_obb[4]):.4f} m")
print(f"P90:     {ajustar_z(percentis_obb[5]):.4f} m")
print(f"máximo:  {ajustar_z(percentis_obb[6]):.4f} m")

print()
print("Perfil ao longo do eixo maior da OBB:")
print(
    resultado[
        [
            "ponto",
            "fracao_eixo_maior",
            "x_px",
            "y_px",
            "z_p10_m",
            "z_mediana_m",
            "z_p90_m",
            "distancia_mediana_m",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print()
print("CSV salvo em:", SAIDA_CSV)
print("Imagem de conferência salva em:", SAIDA_IMAGEM)
