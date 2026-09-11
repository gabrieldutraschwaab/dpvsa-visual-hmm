from pathlib import Path
import base64
import json

import pandas as pd
import requests


RAIZ = Path(__file__).resolve().parents[1]
PASTA_IMAGENS = RAIZ.parent / "api-digital-twin" / "test_images"
PASTA_JSON = RAIZ / "dados" / "respostas_api"
PASTA_PROCESSADOS = RAIZ / "dados" / "processados"
PASTA_RESULTADOS = RAIZ / "resultados"

IMAGENS = ("img1.jpeg", "img5.jpeg")
URL_API = "http://127.0.0.1:8000/process_frame"

CAMERA = {
    "fx": 1268.9369,
    "fy": 1272.6680,
    "cx": 457.3792,
    "cy": 841.9177,
}

IMU = {
    "pos_x": 0.0,
    "pos_y": 0.0,
    "pos_z": 0.0,
    "rot_x": 0.0,
    "rot_y": 0.0,
    "rot_z": 0.0,
    "rot_w": 1.0,
}


def salvar_imagem_base64(conteudo, caminho):
    if conteudo:
        caminho.write_bytes(base64.b64decode(conteudo))


PASTA_JSON.mkdir(parents=True, exist_ok=True)
PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

registros = []

for nome_imagem in IMAGENS:
    caminho_imagem = PASTA_IMAGENS / nome_imagem

    if not caminho_imagem.is_file():
        raise FileNotFoundError(
            f"Imagem não encontrada: {caminho_imagem}"
        )

    imagem_base64 = base64.b64encode(
        caminho_imagem.read_bytes()
    ).decode("utf-8")

    payload = {
        "image_base64": imagem_base64,
        "debug_mode": True,
        "camera": CAMERA,
        "imu": IMU,
    }

    print(f"Processando {nome_imagem}...")

    resposta = requests.post(
        URL_API,
        json=payload,
        timeout=300,
    )

    print("Status HTTP:", resposta.status_code)

    if resposta.status_code != 200:
        print(resposta.text)
        raise RuntimeError(
            f"Falha ao processar {nome_imagem}"
        )

    dados = resposta.json()

    imagem_anotada = dados.pop(
        "image_result_base64",
        None,
    )
    mapa_profundidade = dados.pop(
        "depth_map_base64",
        None,
    )

    nome_base = Path(nome_imagem).stem

    caminho_json = PASTA_JSON / f"{nome_base}.json"
    caminho_json.write_text(
        json.dumps(
            dados,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    salvar_imagem_base64(
        imagem_anotada,
        PASTA_RESULTADOS / f"{nome_base}_deteccoes.jpg",
    )
    salvar_imagem_base64(
        mapa_profundidade,
        PASTA_RESULTADOS / f"{nome_base}_profundidade.jpg",
    )

    ativos = dados.get("assets_to_render", [])

    for numero, ativo in enumerate(ativos, start=1):
        registros.append(
            {
                "imagem": nome_imagem,
                "deteccao": numero,
                "classe": ativo["asset_id"],
                "camera_x": ativo["camera_x"],
                "camera_y": ativo["camera_y"],
                "camera_z": ativo["camera_z"],
                "distancia_camera": ativo["distance_camera"],
                "pos_x": ativo["pos_x"],
                "pos_y": ativo["pos_y"],
                "pos_z": ativo["pos_z"],
                "is_anchor": ativo["is_anchor"],
            }
        )

    print("Detecções:", len(ativos))
    print("JSON salvo em:", caminho_json)
    print()

tabela = pd.DataFrame(registros)

caminho_csv = (
    PASTA_PROCESSADOS
    / "deteccoes_imagens_controladas.csv"
)

tabela.to_csv(
    caminho_csv,
    index=False,
)

print("Tabela salva em:", caminho_csv)
print()
print(
    tabela[
        [
            "imagem",
            "deteccao",
            "classe",
            "camera_z",
            "distancia_camera",
        ]
    ].to_string(index=False)
)
