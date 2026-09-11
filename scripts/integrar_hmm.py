#!/usr/bin/env python3
"""Prepara observações visuais globais e executa o núcleo do HMM existente."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EIXOS = ["x", "y", "z"]
COLUNAS_PREVISOES = {
    "imagem",
    "deteccao",
    "classe",
    "camera_x_p10_m",
    "camera_y_p10_m",
    "camera_z_p10_m",
}
COLUNAS_POSES = {
    "imagem",
    "pos_x_m",
    "pos_y_m",
    "pos_z_m",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_w",
}
COLUNAS_SEQUENCIA = {"sequencia_id", "t", "imagem", "deteccao"}
COLUNAS_ATIVOS = {
    "identidade",
    "classe",
    "pos_x_m",
    "pos_y_m",
    "pos_z_m",
}


def exigir_colunas(tabela: pd.DataFrame, esperadas: set[str], nome: str) -> None:
    ausentes = sorted(esperadas - set(tabela.columns))
    if ausentes:
        raise ValueError(f"Colunas ausentes em {nome}: {ausentes}")


def quaternion_para_matriz(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Replica a convenção (x, y, z, w) da API, rejeitando quaternion nulo."""
    q = np.asarray([qx, qy, qz, qw], dtype=float)
    if not np.isfinite(q).all():
        raise ValueError("Quaternion contém valor não finito.")

    norma = float(np.linalg.norm(q))
    if norma <= 1e-12:
        raise ValueError("Quaternion nulo não define uma orientação.")

    qx, qy, qz, qw = q / norma
    return np.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx**2 + qy**2),
            ],
        ],
        dtype=float,
    )


def transformar_ponto(
    ponto_camera: np.ndarray,
    translacao: np.ndarray,
    quaternion: np.ndarray,
) -> np.ndarray:
    rotacao = quaternion_para_matriz(*quaternion)
    return rotacao @ np.asarray(ponto_camera, dtype=float) + np.asarray(
        translacao, dtype=float
    )


def preparar_observacoes(
    previsoes: pd.DataFrame,
    poses: pd.DataFrame,
    sequencia: pd.DataFrame,
) -> pd.DataFrame:
    exigir_colunas(previsoes, COLUNAS_PREVISOES, "previsões")
    exigir_colunas(poses, COLUNAS_POSES, "poses")
    exigir_colunas(sequencia, COLUNAS_SEQUENCIA, "sequência")

    if previsoes.duplicated(["imagem", "deteccao"]).any():
        raise ValueError("Há previsões duplicadas por imagem e detecção.")
    if poses.duplicated(["imagem"]).any():
        raise ValueError("Cada imagem deve possuir exatamente uma pose.")
    if sequencia.duplicated(["sequencia_id", "t"]).any():
        raise ValueError("Cada instante deve ter uma única observação-alvo.")

    tabela = sequencia.merge(
        previsoes,
        on=["imagem", "deteccao"],
        how="left",
        validate="many_to_one",
        indicator="_previsao",
    )
    faltantes = tabela.loc[
        tabela["_previsao"] != "both", ["imagem", "deteccao"]
    ]
    if not faltantes.empty:
        raise ValueError(
            "Detecções da sequência sem previsão:\n"
            + faltantes.to_string(index=False)
        )
    tabela = tabela.drop(columns="_previsao")

    tabela = tabela.merge(
        poses,
        on="imagem",
        how="left",
        validate="many_to_one",
        indicator="_pose",
    )
    sem_pose = tabela.loc[tabela["_pose"] != "both", "imagem"].unique()
    if len(sem_pose):
        raise ValueError(f"Imagens sem pose: {sorted(sem_pose.tolist())}")
    tabela = tabela.drop(columns="_pose")

    numericas = [
        "t",
        "camera_x_p10_m",
        "camera_y_p10_m",
        "camera_z_p10_m",
        "pos_x_m",
        "pos_y_m",
        "pos_z_m",
        "rot_x",
        "rot_y",
        "rot_z",
        "rot_w",
    ]
    for coluna in numericas:
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="raise")
    if not np.isfinite(tabela[numericas].to_numpy(dtype=float)).all():
        raise ValueError("As observações ou poses contêm valores não finitos.")

    pontos_globais = []
    for registro in tabela.itertuples(index=False):
        ponto_camera = np.array(
            [
                registro.camera_x_p10_m,
                registro.camera_y_p10_m,
                registro.camera_z_p10_m,
            ]
        )
        translacao = np.array(
            [registro.pos_x_m, registro.pos_y_m, registro.pos_z_m]
        )
        quaternion = np.array(
            [registro.rot_x, registro.rot_y, registro.rot_z, registro.rot_w]
        )
        pontos_globais.append(
            transformar_ponto(ponto_camera, translacao, quaternion)
        )

    pontos_globais = np.vstack(pontos_globais)
    tabela["x_observado"] = pontos_globais[:, 0]
    tabela["y_observado"] = pontos_globais[:, 1]
    tabela["z_observado"] = pontos_globais[:, 2]

    colunas = [
        "sequencia_id",
        "t",
        "imagem",
        "deteccao",
        "classe",
    ]
    if "ativo_real" in tabela.columns:
        colunas.append("ativo_real")
    colunas += [
        "x_observado",
        "y_observado",
        "z_observado",
        "camera_x_p10_m",
        "camera_y_p10_m",
        "camera_z_p10_m",
        "pos_x_m",
        "pos_y_m",
        "pos_z_m",
        "rot_x",
        "rot_y",
        "rot_z",
        "rot_w",
    ]
    if "amplitude_p90_p10_m" in tabela.columns:
        colunas.append("amplitude_p90_p10_m")

    return tabela.loc[:, colunas].sort_values(
        ["sequencia_id", "t"], ignore_index=True
    )


def carregar_funcoes_hmm(caminho_notebook: Path) -> dict[str, object]:
    """Extrai somente as funções matemáticas do notebook HMM, sem executá-lo."""
    nomes = {
        "calcular_distancia_mahalanobis",
        "calcular_probabilidades_emissao",
        "construir_matriz_emissoes",
        "executar_forward",
        "converter_para_log",
        "executar_viterbi",
    }
    notebook = json.loads(caminho_notebook.read_text(encoding="utf-8"))
    definicoes: dict[str, ast.FunctionDef] = {}

    for celula in notebook.get("cells", []):
        if celula.get("cell_type") != "code":
            continue
        fonte = "".join(celula.get("source", []))
        try:
            arvore = ast.parse(fonte)
        except SyntaxError:
            continue
        for item in arvore.body:
            if isinstance(item, ast.FunctionDef) and item.name in nomes:
                definicoes[item.name] = item

    ausentes = sorted(nomes - set(definicoes))
    if ausentes:
        raise RuntimeError(f"Funções ausentes no notebook HMM: {ausentes}")

    namespace: dict[str, object] = {"np": np, "pd": pd}
    for nome in (
        "calcular_distancia_mahalanobis",
        "calcular_probabilidades_emissao",
        "construir_matriz_emissoes",
        "executar_forward",
        "converter_para_log",
        "executar_viterbi",
    ):
        modulo = ast.Module(body=[definicoes[nome]], type_ignores=[])
        ast.fix_missing_locations(modulo)
        exec(compile(modulo, str(caminho_notebook), "exec"), namespace)

    return namespace


def carregar_covariancia(caminho: Path) -> np.ndarray:
    tabela = pd.read_csv(caminho, index_col=0)
    try:
        matriz = tabela.loc[EIXOS, EIXOS].to_numpy(dtype=float)
    except KeyError as erro:
        raise ValueError("A covariância deve possuir linhas e colunas x, y e z.") from erro
    if not np.isfinite(matriz).all() or not np.allclose(matriz, matriz.T):
        raise ValueError("A matriz de covariância deve ser finita e simétrica.")
    if np.any(np.linalg.eigvalsh(matriz) <= 0):
        raise ValueError("A matriz de covariância deve ser definida positiva.")
    return matriz


def carregar_ativos(caminho: Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    tabela = pd.read_csv(caminho)
    exigir_colunas(tabela, COLUNAS_ATIVOS, "ativos")
    if tabela["identidade"].duplicated().any():
        raise ValueError("As identidades dos ativos devem ser únicas.")
    if tabela[["identidade", "classe"]].isna().any().any():
        raise ValueError("Identidade e classe dos ativos não podem ser nulas.")
    coordenadas = tabela[["pos_x_m", "pos_y_m", "pos_z_m"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(coordenadas.to_numpy(dtype=float)).all():
        raise ValueError("As posições dos ativos devem ser finitas.")
    ativos = {
        str(linha.identidade): {
            "classe": str(linha.classe),
            "posicao_3d": (
                float(linha.pos_x_m),
                float(linha.pos_y_m),
                float(linha.pos_z_m),
            ),
        }
        for linha in tabela.itertuples(index=False)
    }
    return tabela, ativos


def carregar_transicoes(caminho: Path, identidades: list[str]) -> pd.DataFrame:
    matriz = pd.read_csv(caminho, index_col=0)
    matriz.index = matriz.index.astype(str)
    matriz.columns = matriz.columns.astype(str)
    try:
        matriz = matriz.loc[identidades, identidades].astype(float)
    except KeyError as erro:
        raise ValueError("A matriz de transição não cobre todos os ativos.") from erro
    valores = matriz.to_numpy()
    if not np.isfinite(valores).all() or (valores < 0).any():
        raise ValueError("As transições devem ser probabilidades finitas e não negativas.")
    if not np.allclose(valores.sum(axis=1), 1.0):
        raise ValueError("Cada linha da matriz de transição deve somar 1.")
    return matriz


def restringir_emissoes_por_classe(
    emissoes: pd.DataFrame,
    observacoes: pd.DataFrame,
    ativos_tabela: pd.DataFrame,
) -> pd.DataFrame:
    classes_ativos = ativos_tabela.set_index("identidade")["classe"].astype(str)
    classes_observadas = observacoes["classe"].astype(str).to_numpy()
    mascara = np.vstack(
        [
            classes_ativos.reindex(emissoes.columns).eq(classe).to_numpy()
            for classe in classes_observadas
        ]
    )
    sem_candidato = ~mascara.any(axis=1)
    if sem_candidato.any():
        classes = sorted(set(classes_observadas[sem_candidato].tolist()))
        raise ValueError(f"Classes observadas sem ativos candidatos: {classes}")

    valores = emissoes.to_numpy(dtype=float) * mascara
    valores /= valores.sum(axis=1, keepdims=True)
    return pd.DataFrame(valores, index=emissoes.index, columns=emissoes.columns)


def executar_hmm(
    observacoes: pd.DataFrame,
    ativos_tabela: pd.DataFrame,
    ativos: dict[str, dict],
    transicoes: pd.DataFrame,
    covariancia: np.ndarray,
    funcoes: dict[str, object],
) -> pd.DataFrame:
    if observacoes.empty:
        raise ValueError("Não há observações para executar o HMM.")
    identidades = ativos_tabela["identidade"].astype(str).tolist()
    distribuicao_inicial = pd.Series(1.0 / len(identidades), index=identidades)
    partes = []

    for sequencia_id, grupo in observacoes.groupby("sequencia_id", sort=False):
        grupo = grupo.sort_values("t").reset_index(drop=True)
        if grupo["t"].duplicated().any():
            raise ValueError(f"Instantes duplicados na sequência {sequencia_id!r}.")

        emissoes = funcoes["construir_matriz_emissoes"](
            observacoes=grupo,
            ativos=ativos,
            covariancia=covariancia,
            identidades=identidades,
        )
        emissoes = restringir_emissoes_por_classe(
            emissoes,
            grupo,
            ativos_tabela,
        )
        posteriores = funcoes["executar_forward"](
            matriz_emissoes=emissoes,
            matriz_transicao=transicoes,
            distribuicao_inicial=distribuicao_inicial,
        )
        viterbi = funcoes["executar_viterbi"](
            matriz_emissoes=emissoes,
            matriz_transicao=transicoes,
            distribuicao_inicial=distribuicao_inicial,
        )
        if isinstance(viterbi, pd.DataFrame):
            if "predicao_viterbi" in viterbi.columns:
                predicao_viterbi = viterbi["predicao_viterbi"].to_numpy()
            elif viterbi.shape[1] == 1:
                predicao_viterbi = viterbi.iloc[:, 0].to_numpy()
            else:
                raise RuntimeError("Formato tabular inesperado retornado pelo Viterbi.")
        elif isinstance(viterbi, pd.Series):
            predicao_viterbi = viterbi.to_numpy()
        else:
            predicao_viterbi = np.asarray(viterbi)
        if len(predicao_viterbi) != len(grupo):
            raise RuntimeError("O Viterbi retornou quantidade inesperada de estados.")

        saida = grupo.copy()
        saida["predicao_espacial"] = emissoes.idxmax(axis=1).to_numpy()
        saida["confianca_espacial"] = emissoes.max(axis=1).to_numpy()
        saida["predicao_forward"] = posteriores.idxmax(axis=1).to_numpy()
        saida["confianca_forward"] = posteriores.max(axis=1).to_numpy()
        saida["predicao_viterbi"] = predicao_viterbi
        partes.append(saida)

    return pd.concat(partes, ignore_index=True)


def autoteste(caminho_notebook: Path) -> None:
    previsoes = pd.DataFrame(
        {
            "imagem": ["f1", "f2", "f3"],
            "deteccao": [1, 1, 1],
            "classe": ["valvula"] * 3,
            "camera_x_p10_m": [0.1, 0.2, 0.0],
            "camera_y_p10_m": [0.0, 0.0, 0.0],
            "camera_z_p10_m": [1.0, 1.0, 1.0],
        }
    )
    poses = pd.DataFrame(
        {
            "imagem": ["f1", "f2", "f3"],
            "pos_x_m": [0.0, 0.0, 4.0],
            "pos_y_m": [0.0, 0.0, 0.0],
            "pos_z_m": [0.0, 0.0, 0.0],
            "rot_x": [0.0, 0.0, 0.0],
            "rot_y": [0.0, 0.0, 0.0],
            "rot_z": [0.0, 0.0, 0.0],
            "rot_w": [1.0, 1.0, 1.0],
        }
    )
    sequencia = pd.DataFrame(
        {
            "sequencia_id": ["teste"] * 3,
            "t": [1, 2, 3],
            "imagem": ["f1", "f2", "f3"],
            "deteccao": [1, 1, 1],
            "ativo_real": ["A", "A", "B"],
        }
    )
    observacoes = preparar_observacoes(previsoes, poses, sequencia)
    esperados = np.array([[0.1, 0.0, 1.0], [0.2, 0.0, 1.0], [4.0, 0.0, 1.0]])
    obtidos = observacoes[["x_observado", "y_observado", "z_observado"]].to_numpy()
    assert np.allclose(obtidos, esperados)

    ativos_tabela = pd.DataFrame(
        {
            "identidade": ["A", "B"],
            "classe": ["valvula", "valvula"],
            "pos_x_m": [0.0, 4.0],
            "pos_y_m": [0.0, 0.0],
            "pos_z_m": [1.0, 1.0],
        }
    )
    ativos = {
        "A": {"classe": "valvula", "posicao_3d": (0.0, 0.0, 1.0)},
        "B": {"classe": "valvula", "posicao_3d": (4.0, 0.0, 1.0)},
    }
    transicoes = pd.DataFrame(
        [[0.8, 0.2], [0.2, 0.8]], index=["A", "B"], columns=["A", "B"]
    )
    covariancia = np.diag([0.25**2, 0.25**2, 0.25**2])
    funcoes = carregar_funcoes_hmm(caminho_notebook)
    resultado = executar_hmm(
        observacoes,
        ativos_tabela,
        ativos,
        transicoes,
        covariancia,
        funcoes,
    )
    esperado = ["A", "A", "B"]
    assert resultado["predicao_espacial"].tolist() == esperado
    assert resultado["predicao_forward"].tolist() == esperado
    assert resultado["predicao_viterbi"].tolist() == esperado

    angulo = np.pi / 2
    girado = transformar_ponto(
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 0.0, np.sin(angulo / 2), np.cos(angulo / 2)]),
    )
    assert np.allclose(girado, [1.0, 3.0, 3.0])
    print("Autoteste concluído: transformação, adaptador e HMM compatíveis.")
    hash_notebook = hashlib.sha256(caminho_notebook.read_bytes()).hexdigest()
    print("SHA-256 do notebook HMM:", hash_notebook)
    colunas_resultado = [
        "t",
        "ativo_real",
        "predicao_espacial",
        "predicao_forward",
        "predicao_viterbi",
    ]
    print(resultado[colunas_resultado].to_string(index=False))


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook-hmm", type=Path, required=True)
    parser.add_argument("--autoteste", action="store_true")
    parser.add_argument("--previsoes", type=Path)
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--sequencia", type=Path)
    parser.add_argument("--ativos", type=Path)
    parser.add_argument("--transicoes", type=Path)
    parser.add_argument("--covariancia", type=Path)
    parser.add_argument("--saida-observacoes", type=Path)
    parser.add_argument("--saida-resultados", type=Path)
    return parser


def main() -> None:
    argumentos = construir_parser().parse_args()
    if argumentos.autoteste:
        autoteste(argumentos.notebook_hmm)
        return

    obrigatorios = [
        "previsoes",
        "poses",
        "sequencia",
        "ativos",
        "transicoes",
        "covariancia",
        "saida_observacoes",
        "saida_resultados",
    ]
    ausentes = [nome for nome in obrigatorios if getattr(argumentos, nome) is None]
    if ausentes:
        raise SystemExit(f"Argumentos obrigatórios ausentes: {ausentes}")

    previsoes = pd.read_csv(argumentos.previsoes)
    poses = pd.read_csv(argumentos.poses)
    sequencia = pd.read_csv(argumentos.sequencia)
    observacoes = preparar_observacoes(previsoes, poses, sequencia)
    ativos_tabela, ativos = carregar_ativos(argumentos.ativos)
    identidades = ativos_tabela["identidade"].astype(str).tolist()
    transicoes = carregar_transicoes(argumentos.transicoes, identidades)
    covariancia = carregar_covariancia(argumentos.covariancia)
    funcoes = carregar_funcoes_hmm(argumentos.notebook_hmm)
    resultados = executar_hmm(
        observacoes,
        ativos_tabela,
        ativos,
        transicoes,
        covariancia,
        funcoes,
    )

    argumentos.saida_observacoes.parent.mkdir(parents=True, exist_ok=True)
    argumentos.saida_resultados.parent.mkdir(parents=True, exist_ok=True)
    observacoes.to_csv(argumentos.saida_observacoes, index=False)
    resultados.to_csv(argumentos.saida_resultados, index=False)
    print(f"Observações salvas em: {argumentos.saida_observacoes.resolve()}")
    print(f"Resultados salvos em: {argumentos.saida_resultados.resolve()}")


if __name__ == "__main__":
    main()
