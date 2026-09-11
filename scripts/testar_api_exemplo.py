from pathlib import Path
import base64
import json

import requests


RAIZ = Path(__file__).resolve().parents[1]
IMAGEM = (
    RAIZ.parent
    / "api-digital-twin"
    / "test_images"
    / "img5.jpeg"
)
SAIDA = (
    RAIZ
    / "dados"
    / "respostas_api"
    / "exemplo_img5.json"
)

imagem_base64 = base64.b64encode(
    IMAGEM.read_bytes()
).decode("utf-8")

payload = {
    "image_base64": imagem_base64,
    "debug_mode": False,
    "camera": {
        "fx": 1268.9369,
        "fy": 1272.6680,
        "cx": 457.3792,
        "cy": 841.9177,
    },
    "imu": {
        "pos_x": 0.0,
        "pos_y": 0.0,
        "pos_z": 0.0,
        "rot_x": 0.0,
        "rot_y": 0.0,
        "rot_z": 0.0,
        "rot_w": 1.0,
    },
}

resposta = requests.post(
    "http://127.0.0.1:8000/process_frame",
    json=payload,
    timeout=300,
)

print("Status HTTP:", resposta.status_code)

if resposta.status_code != 200:
    print(resposta.text)
    raise SystemExit(1)

dados = resposta.json()

SAIDA.write_text(
    json.dumps(
        dados,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

ativos = dados.get("assets_to_render", [])

print("Imagem:", IMAGEM)
print("Detecções:", len(ativos))
print("Resposta salva em:", SAIDA)

for numero, ativo in enumerate(ativos, start=1):
    print()
    print("Detecção", numero)
    print("Classe:", ativo["asset_id"])
    print(
        "Câmera:",
        ativo["camera_x"],
        ativo["camera_y"],
        ativo["camera_z"],
    )
    print(
        "Distância da câmera:",
        ativo["distance_camera"],
    )
    print(
        "Global:",
        ativo["pos_x"],
        ativo["pos_y"],
        ativo["pos_z"],
    )
