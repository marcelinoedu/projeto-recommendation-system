# Recomendacao de Filmes por Fatoracao de Matrizes

Este projeto implementa e compara modelos de recomendacao de filmes usando dados reais do MovieLens Latest Small. O problema e modelado como uma matriz usuario-filme esparsa, em que cada linha representa um usuario, cada coluna representa um filme e cada entrada conhecida representa uma nota dada pelo usuario.

O objetivo e responder:

> Com base nos filmes que um usuario ja avaliou, quais outros filmes ele provavelmente gostaria de assistir?

Para isso, foram comparadas tres abordagens:

1. Popularidade
2. SVD truncado
3. Matrix Factorization com embeddings

O projeto segue a ideia apresentada pela referencia do Google Developers sobre sistemas de recomendacao, especialmente a formulacao de recomendacao como fatoracao de matrizes e aprendizado de fatores latentes.

Referencia principal:

- Google Developers. *Recommendation Systems*. Disponivel em: https://developers.google.com/machine-learning/recommendation?hl=pt-br

---

## 1. Dataset utilizado

O dataset escolhido foi o **MovieLens Latest Small**.

Resumo dos dados usados:

| Item | Quantidade |
| --- | ---: |
| Usuarios | 610 |
| Filmes | 9.724 |
| Avaliacoes | 100.836 |
| Avaliacoes de treino | 80.672 |
| Avaliacoes de teste | 20.164 |
| Media global no treino | 3,5034 |

Arquivos principais:

- `data/ml_data/ratings.csv`
- `data/ml_data/movies.csv`

A distribuicao das notas tambem foi calculada:

| Nota | Quantidade |
| ---: | ---: |
| 0.5 | 1.370 |
| 1.0 | 2.811 |
| 1.5 | 1.791 |
| 2.0 | 7.551 |
| 2.5 | 5.550 |
| 3.0 | 20.047 |
| 3.5 | 13.136 |
| 4.0 | 26.818 |
| 4.5 | 8.551 |
| 5.0 | 13.211 |

---

## 2. Modelagem por Algebra Linear

A partir das avaliacoes, construimos uma matriz usuario-filme:

| Usuario / Filme | Filme A | Filme B | Filme C | Filme D |
| --- | ---: | ---: | ---: | ---: |
| Usuario 1 | 5 | ? | 3 | ? |
| Usuario 2 | ? | 4 | ? | 2 |
| Usuario 3 | 4 | ? | ? | 5 |

O simbolo `?` representa uma nota desconhecida. O problema de recomendacao e estimar essas entradas faltantes de forma que as maiores previsoes indiquem filmes recomendados para cada usuario.

---

## 3. Modelos comparados

### 3.1 Popularidade

O modelo de popularidade recomenda filmes com melhores avaliacoes medias no conjunto de treino. Ele funciona como baseline, pois nao personaliza fortemente as recomendacoes por usuario.

A pontuacao de um filme e dada por:

```text
score(i) = (1 / N_i) * soma r_ui
```

Onde:

- `i` e o filme;
- `u` e o usuario;
- `r_ui` e a nota dada pelo usuario `u` ao filme `i`;
- `N_i` e o numero de avaliacoes do filme `i`.

### 3.2 SVD truncado

O SVD decompoe a matriz de avaliacoes em tres matrizes:

```text
R ~= U Sigma V^T
```

Como a matriz original e esparsa, foi necessario tratar valores faltantes antes da decomposicao. A estrategia usada foi centralizar as notas por usuario, preencher ausencias com zero e reconstruir a matriz com apenas os primeiros `k` fatores:

```text
R_hat = U_k Sigma_k V_k^T
```

O valor `k` controla a quantidade de fatores latentes usados na aproximacao.

### 3.3 Matrix Factorization com embeddings

A Matrix Factorization representa usuarios e filmes por vetores latentes. A previsao da nota e:

```text
r_hat_ui = mu + b_u + b_i + p_u^T q_i
```

Onde:

- `mu` e a media global das notas;
- `b_u` e o vies do usuario;
- `b_i` e o vies do filme;
- `p_u` e o embedding do usuario;
- `q_i` e o embedding do filme;
- `p_u^T q_i` mede a compatibilidade entre usuario e filme.

O modelo foi treinado minimizando erro quadratico com regularizacao:

```text
L = soma (r_ui - r_hat_ui)^2 + lambda (||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)
```

---

## 4. Avaliacao

Os modelos foram avaliados de duas formas:

1. **Previsao de nota**, usando RMSE e MAE.
2. **Qualidade das recomendacoes Top-10**, usando Precision@10 e Recall@10.

Um filme foi considerado relevante quando o usuario deu nota maior ou igual a `4`.

### Metricas de erro

```text
RMSE = sqrt((1 / |T|) * soma (r_ui - r_hat_ui)^2)
MAE = (1 / |T|) * soma |r_ui - r_hat_ui|
```

Quanto menor o RMSE e o MAE, melhor.

### Metricas Top-10

```text
Precision@10 = relevantes recomendados / 10
Recall@10 = relevantes recomendados / relevantes no teste
```

Quanto maiores Precision@10 e Recall@10, melhor.

---

## 5. Resultados obtidos

### 5.1 Comparacao de erro

| Modelo | RMSE | MAE |
| --- | ---: | ---: |
| Popularidade | 0,9667 | 0,7525 |
| SVD k=10 | 0,9140 | 0,7076 |
| Matrix Factorization k=5 | 0,9139 | 0,7000 |

Em erro de previsao, a Matrix Factorization teve o melhor resultado, mas a diferenca para o SVD foi muito pequena. Ambos superaram o modelo de popularidade.

### 5.2 Comparacao Top-10

| Modelo | Precision@10 | Recall@10 | Usuarios avaliados |
| --- | ---: | ---: | ---: |
| Popularidade | 0,0779 | 0,0614 | 602 |
| SVD k=10 | 0,1271 | 0,0972 | 602 |
| Matrix Factorization k=5 | 0,0211 | 0,0184 | 602 |

Nas recomendacoes Top-10, o SVD teve o melhor desempenho. Esse resultado mostra que prever notas com baixo erro nao garante, necessariamente, melhor ordenacao dos filmes recomendados.

### 5.3 Efeito do numero de fatores `k`

| k | RMSE SVD | MAE SVD | Precision@10 SVD | Recall@10 SVD | RMSE MF | MAE MF | Precision@10 MF | Recall@10 MF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0,9157 | 0,7089 | 0,1196 | 0,0900 | 0,9139 | 0,7000 | 0,0211 | 0,0184 |
| 10 | 0,9140 | 0,7076 | 0,1271 | 0,0972 | 0,9332 | 0,7154 | 0,0186 | 0,0143 |
| 20 | 0,9174 | 0,7099 | 0,1208 | 0,0956 | 0,9522 | 0,7328 | 0,0281 | 0,0193 |
| 50 | 0,9258 | 0,7173 | 0,1106 | 0,1077 | 0,9594 | 0,7404 | 0,0407 | 0,0309 |
| 100 | 0,9342 | 0,7261 | 0,0817 | 0,0885 | 0,9166 | 0,7093 | 0,0555 | 0,0407 |

Para o SVD, o melhor RMSE ocorreu com `k=10`. Para a Matrix Factorization, o melhor RMSE ocorreu com `k=5`. Valores maiores de `k` nem sempre melhoraram o resultado, o que sugere risco de aumento de complexidade sem ganho real de generalizacao.

### 5.4 Exemplo de recomendacao

Foi escolhido o usuario `414`.

Alguns filmes avaliados com nota `5.0` por esse usuario no treino:

| Filme | Nota |
| --- | ---: |
| Cyrano de Bergerac (1990) | 5.0 |
| Fantasia (1940) | 5.0 |
| Terminator, The (1984) | 5.0 |
| Graduate, The (1967) | 5.0 |
| Chinatown (1974) | 5.0 |
| Inception (2010) | 5.0 |

Top-5 recomendacoes por popularidade:

| Posicao | Filme recomendado | Score |
| ---: | --- | ---: |
| 1 | Usual Suspects, The (1995) | 4,2181 |
| 2 | Godfather: Part II, The (1974) | 4,1867 |
| 3 | Streetcar Named Desire, A (1951) | 4,1564 |
| 4 | Snatch (2000) | 4,1357 |
| 5 | Raiders of the Lost Ark (1981) | 4,1215 |

Top-5 recomendacoes por SVD:

| Posicao | Filme recomendado | Score |
| ---: | --- | ---: |
| 1 | Big (1988) | 3,6050 |
| 2 | Patriot, The (2000) | 3,5799 |
| 3 | Royal Tenenbaums, The (2001) | 3,5736 |
| 4 | Home Alone 2: Lost in New York (1992) | 3,5684 |
| 5 | Spice World (1997) | 3,5679 |

Top-5 recomendacoes por Matrix Factorization:

| Posicao | Filme recomendado | Score |
| ---: | --- | ---: |
| 1 | Jonah Who Will Be 25 in the Year 2000 (1976) | 5,0000 |
| 2 | Monty Python and the Holy Grail (1975) | 5,0000 |
| 3 | Battle of Algiers, The (1966) | 5,0000 |
| 4 | Lives of Others, The (2006) | 5,0000 |
| 5 | Gallipoli (1981) | 5,0000 |

---

## 6. Interpretacao no problema real

No contexto de uma plataforma de filmes, os resultados indicam que a escolha do melhor modelo depende do objetivo.

Se o objetivo for prever a nota que um usuario daria a um filme, a Matrix Factorization foi ligeiramente melhor. Porem, se o objetivo for montar uma lista Top-10 de recomendacoes relevantes, o SVD foi superior nos experimentos realizados.

Isso e importante porque um sistema real de recomendacao normalmente precisa ordenar itens, nao apenas prever notas isoladas. Assim, os resultados sugerem que otimizar RMSE/MAE pode nao ser suficiente para gerar boas recomendacoes finais.

---

## 7. Limitacoes

Algumas limitacoes observadas:

1. O dataset e pequeno em comparacao com sistemas reais de streaming.
2. Os modelos usam apenas notas, sem considerar genero, elenco, tags ou sinopse.
3. Usuarios com poucas avaliacoes continuam sendo mais dificeis de modelar.
4. Filmes com poucas avaliacoes podem receber estimativas menos confiaveis.
5. O SVD precisa preencher artificialmente valores faltantes antes da decomposicao.
6. A Matrix Factorization teve bom RMSE, mas baixo Precision@10, mostrando desalinhamento entre erro de previsao e qualidade de ranking.
7. As previsoes da Matrix Factorization foram limitadas ao intervalo de notas, gerando varios scores iguais a `5.0` nas recomendacoes e possiveis empates no ranking.

---

## 8. Estrutura do projeto

```text
projeto-recommendation-system/
|-- data/
|   |-- ml-latest-small.zip
|   `-- ml_data/
|       |-- ratings.csv
|       |-- movies.csv
|       |-- tags.csv
|       `-- links.csv
|-- notebooks/
|   |-- treinamento.ipynb
|   `-- relatorio.ipynb
|-- results/
|   |-- metrics_rating.csv
|   |-- metrics_top10.csv
|   |-- experiment_k.csv
|   |-- svd_k_results.csv
|   |-- mf_k_results.csv
|   |-- recommendation_examples.csv
|   |-- recommendations_popularity.csv
|   |-- recommendations_svd.csv
|   |-- recommendations_mf.csv
|   |-- summary.json
|   `-- best_mf_model.pt
|-- main.py
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

---

## 9. Como executar

Este projeto usa `pyproject.toml` e `uv.lock` para gerenciar dependencias.

Instalar dependencias:

```bash
uv sync
```

Executar os notebooks com Jupyter, caso ele ja esteja instalado:

```bash
uv run jupyter notebook notebooks/treinamento.ipynb
uv run jupyter notebook notebooks/relatorio.ipynb
```

Se o comando `jupyter` nao estiver disponivel, executar com uma dependencia temporaria:

```bash
uv run --with jupyter jupyter notebook notebooks/treinamento.ipynb
uv run --with jupyter jupyter notebook notebooks/relatorio.ipynb
```

O notebook `treinamento.ipynb` contem o fluxo principal de treino, avaliacao e geracao dos arquivos em `results/`. O notebook `relatorio.ipynb` organiza os resultados para analise.

Observacao: o arquivo `main.py` ainda nao reproduz o pipeline completo; ele esta apenas como ponto de entrada minimo do projeto.

---

## 10. Status do projeto

| Etapa | Status |
| --- | --- |
| Baixar dataset | Concluido |
| Carregar dados | Concluido |
| Explorar dados | Concluido |
| Criar matriz usuario-filme | Concluido |
| Separar treino e teste | Concluido |
| Implementar popularidade | Concluido |
| Implementar SVD | Concluido |
| Implementar Matrix Factorization | Concluido |
| Avaliar RMSE e MAE | Concluido |
| Avaliar Precision@10 e Recall@10 | Concluido |
| Testar diferentes valores de k | Concluido |
| Gerar exemplos de recomendacao | Concluido |
| Gerar graficos finais | Pendente |
| Escrever relatorio final em IEEE | Pendente |

---

## 11. Conclusao

O projeto mostra como um problema real de recomendacao pode ser formulado como um problema de Algebra Linear. A matriz usuario-filme permite representar avaliacoes conhecidas e estimar avaliacoes faltantes por meio de decomposicoes e fatoracoes.

Os resultados mostram que SVD e Matrix Factorization superam a popularidade na previsao de notas. No entanto, para recomendacoes Top-10, o SVD teve desempenho superior, reforcando que a metrica de avaliacao deve estar alinhada ao objetivo real do sistema.
