# Integração estrutural entre visão computacional e HMM

## Objetivo

Estabelecer uma interface reproduzível entre as coordenadas tridimensionais
estimadas pelo pipeline visual e os métodos de associação espacial, Forward e
Viterbi implementados no projeto `HMM_ASSET_DESAMBIGUATION`.

## Interface implementada

O script `scripts/integrar_hmm.py` recebe cinco conjuntos de dados:

1. **Previsões visuais:** identificador da imagem e da detecção, classe e
   coordenadas `camera_x_p10_m`, `camera_y_p10_m` e `camera_z_p10_m`.
2. **Poses da câmera:** uma posição e um quaternion de orientação para cada
   imagem.
3. **Sequência de inspeção:** seleção de exatamente uma detecção-alvo por
   instante de cada sequência.
4. **Cadastro dos ativos:** identidade, classe e posição tridimensional global
   de cada Gêmeo Digital candidato.
5. **Modelo probabilístico:** matriz de covariância posicional e matriz de
   transição entre as identidades.

As observações produzidas para o HMM contêm, entre outros metadados, as colunas
`sequencia_id`, `t`, `x_observado`, `y_observado` e `z_observado`.

Os arquivos-modelo ficam em `dados/integracao_hmm/`. A covariância deve ser uma
matriz 3 × 3 finita, simétrica e definida positiva, rotulada pelos eixos `x`,
`y` e `z`. A matriz de transição deve usar as identidades como linhas e colunas,
conter apenas probabilidades não negativas e ter soma unitária em cada linha.

Uma execução com dados reais seguirá a forma:

```bash
python scripts/integrar_hmm.py \
  --notebook-hmm CAMINHO/01_cenario_sintetico.ipynb \
  --previsoes dados/processados/previsoes_validacao_p10.csv \
  --poses dados/integracao_hmm/poses_camera.csv \
  --sequencia dados/integracao_hmm/sequencia_observacoes.csv \
  --ativos dados/integracao_hmm/ativos_hmm.csv \
  --transicoes dados/integracao_hmm/transicoes_hmm.csv \
  --covariancia dados/integracao_hmm/covariancia_hmm.csv \
  --saida-observacoes dados/processados/observacoes_hmm.csv \
  --saida-resultados resultados/integracao_hmm/associacoes_hmm.csv
```

## Transformação geométrica

A transformação reproduz a convenção utilizada pela API de visão:

```text
p_global = R(qx, qy, qz, qw) @ p_camera + translacao
```

O quaternion é normalizado antes da construção da matriz de rotação. O
adaptador rejeita quaternions nulos, valores não finitos, poses duplicadas e
detecções ausentes.

## Integração com o HMM

Para evitar duplicar ou modificar a implementação do HMM, o adaptador extrai
do notebook-base somente as definições das seguintes funções matemáticas:

- distância de Mahalanobis;
- probabilidades e matriz de emissão;
- recursão Forward;
- conversão para logaritmos;
- Viterbi.

O notebook não é executado integralmente. As emissões são restringidas aos
ativos cuja classe coincide com a classe detectada; dentro de cada classe, a
desambiguação continua sendo determinada pela posição e pelo histórico.

## Verificação realizada

O arquivo `scripts/integrar_hmm.py` foi verificado por compilação e por um
autoteste de integração em memória.

SHA-256 do adaptador:

```text
a13b0c3fcaf302256d9e2ba872a41394f69721b44eba46d280acd70175621b3c
```

SHA-256 do notebook HMM utilizado no autoteste:

```text
479ee3f8efe8a61adc530d22dace7ed70bc5afffbe129d8067ba21ec69e2e32f
```

O cenário do autoteste contém dois ativos visualmente idênticos, `A` e `B`, e
uma sequência verdadeira `A, A, B`. A transformação inclui poses identidade e
uma translação conhecida. Os três métodos produziram a sequência esperada:

| t | Real | Espacial | Forward | Viterbi |
| -: | :--- | :------- | :------ | :------ |
| 1 | A | A | A | A |
| 2 | A | A | A | A |
| 3 | B | B | B | B |

Esse resultado verifica a compatibilidade de software, o contrato de dados, a
transformação geométrica e a chamada das funções originais do HMM.

## Limite da evidência atual

O autoteste não constitui validação ponta a ponta com dados reais. Nas imagens
já processadas, a posição da IMU foi definida como `(0, 0, 0)` e sua orientação
como o quaternion identidade `(0, 0, 0, 1)`. Consequentemente, os campos globais
retornados pela API coincidem com as coordenadas relativas à câmera.

Também não estão disponíveis, para as imagens atuais:

- poses globais medidas da câmera;
- posições globais cadastradas das válvulas reais;
- uma sequência temporal com uma detecção-alvo e identidade de referência em
  cada instante;
- o grafo ou a matriz de transição da instalação real;
- uma matriz de covariância tridimensional estimada experimentalmente.

As seis medidas independentes existentes permitem avaliar o erro escalar de
distância e a ordenação relativa, mas não permitem estimar uma covariância 3D
nem medir o ganho real de Forward ou Viterbi sobre a associação espacial.

## Conclusão

A ponte de software entre visão e HMM está implementada e foi verificada em um
caso sintético controlado. A avaliação real de desambiguação sequencial fica
condicionada à aquisição sincronizada de poses, observações-alvo e referências
globais dos ativos. Até essa coleta, os resultados quantitativos do P10 e do
HMM devem ser relatados como avaliações complementares, e não como uma única
validação ponta a ponta.
