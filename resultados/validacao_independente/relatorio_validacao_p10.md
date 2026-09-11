# Validação independente piloto da amostragem P10

## Objetivo

Avaliar, em imagens não utilizadas na escolha da estratégia, se a
amostragem do percentil 10 da profundidade dentro da OBB melhora a
estimativa de distância em relação à amostragem central empregada
originalmente pela API.

Também foi avaliada a capacidade dos métodos de preservar a ordenação
relativa entre a válvula mais próxima e a mais distante, informação
potencialmente útil para a associação de válvulas aos Gêmeos Digitais
por meio de um HMM.

## Protocolo

A estratégia P10 foi escolhida exploratoriamente utilizando as imagens
`img1.jpeg` e `img5.jpeg`.

A validação utilizou três imagens diferentes:

- `img2.jpeg`;
- `img3.jpeg`;
- `img4.jpeg`.

Foram avaliadas duas válvulas por imagem, totalizando seis medições.

A separação entre previsões e gabarito foi registrada no Git:

| Etapa | Commit |
|---|---|
| Congelamento das previsões P10 | `250d70c` |
| Registro posterior do gabarito | `150f31d` |
| Avaliação dos erros absolutos | `be42fe9` |
| Avaliação da ordenação relativa | `c28c83d` |

As previsões foram geradas e congeladas antes da inclusão das
distâncias reais e das correspondências visuais.

## Resultados de distância absoluta

| Método | Medições | Viés (m) | MAE (m) | RMSE (m) | MAPE (%) | Erro máximo (m) |
|---|---:|---:|---:|---:|---:|---:|
| Centro da API | 6 | 0,3812 | 0,5290 | 0,7208 | 51,8886 | 1,5325 |
| OBB P10 | 6 | 0,1298 | 0,3265 | 0,3784 | 35,2937 | 0,6634 |

O P10 reduziu o MAE em 38,27% e o RMSE em 47,51%. Ele apresentou
menor erro absoluto em quatro das seis medições.

Entretanto, o P10 não foi superior em todos os casos. Nas duas
medições da `img4.jpeg`, a amostragem central apresentou menor erro
absoluto.

## Resultados de ordenação relativa

| Método | Pares | Ordenações corretas | Acurácia | MAE da margem (m) | RMSE da margem (m) |
|---|---:|---:|---:|---:|---:|
| Centro da API | 3 | 3 | 100% | 0,6214 | 0,7526 |
| OBB P10 | 3 | 3 | 100% | 0,3321 | 0,3765 |

Os dois métodos identificaram corretamente, nas três imagens, qual
válvula estava mais próxima e qual estava mais distante.

O P10 reduziu o erro médio da margem de separação, mas os dois métodos
superestimaram as diferenças reais de distância.

Somente o par da `img4.jpeg` contém válvulas da mesma classe detectada.
Nesse par, ambos preservaram a ordenação, porém a amostragem central
estimou a margem com menor erro.

## Interpretação

A validação piloto indica que o P10 pode reduzir erros grandes causados
pela amostragem central em regiões heterogêneas da OBB. Entretanto, a
estratégia desloca sistematicamente a estimativa para profundidades
menores e pode piorar resultados quando a estimativa central já está
subestimada.

A preservação da ordenação próxima/distante sugere que a profundidade
visual pode fornecer uma observação relativa útil ao HMM mesmo quando
a escala métrica absoluta apresenta erro.

## Limitações

- A validação contém apenas seis medições em três imagens.
- Apenas um par pertence à mesma classe detectada.
- Foram medidas somente as válvulas mais próxima e mais distante.
- Ainda não foram avaliadas sequências temporais.
- A calibração geométrica e a distorção da câmera ainda precisam ser
  verificadas experimentalmente.
- Os resultados não demonstram, isoladamente, melhoria na associação
  de Gêmeos Digitais pelo HMM.

## Conclusão

O P10 permanece como uma estratégia candidata, pois melhorou as
métricas agregadas no conjunto independente. Contudo, os resultados
não justificam sua adoção como substituição universal da amostragem
central.

O resultado mais consistente desta etapa é a preservação da ordenação
espacial nos três pares avaliados. A próxima validação deve incluir
mais válvulas da mesma classe, diferentes posições e sequências de
imagens, seguida da comparação da associação do HMM com e sem as
observações visuais.
