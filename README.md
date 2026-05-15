````markdown
# Recomendação de Filmes por Fatoração de Matrizes

Este projeto tem como objetivo criar um sistema simples de recomendação de filmes usando dados reais do MovieLens.

A ideia principal é comparar três formas de recomendar filmes:

1. **Popularidade**
2. **SVD**
3. **Matrix Factorization com embeddings**

O objetivo é verificar qual método consegue prever melhor os filmes que um usuário provavelmente gostaria de assistir através de uma abordagem linear.

---

## 1. Problema

Hoje existem muitos filmes disponíveis em plataformas de streaming. Um usuário dificilmente consegue olhar todos os filmes para escolher o que assistir.

Por isso, sistemas de recomendação tentam responder a seguinte pergunta:

> Com base nos filmes que o usuário já avaliou, quais outros filmes ele provavelmente gostaria?

Neste projeto, vamos usar avaliações reais de usuários para treinar modelos que tentam prever notas ou recomendar filmes.

---

## 2. Dataset utilizado

O dataset escolhido foi o **MovieLens Latest Small**.

Ele contém aproximadamente:

- 100.000 avaliações;
- 9.000 filmes;
- 600 usuários;
- notas de 0.5 até 5.0.

Arquivos principais usados:
ratings.csv
movies.csv


## 3. Ideia geral do projeto

A partir das avaliações, será criada uma matriz usuário-filme.

Cada linha representa um usuário.

Cada coluna representa um filme.

Cada valor representa a nota que o usuário deu para aquele filme.

Exemplo:

| Usuário / Filme | Filme A | Filme B | Filme C | Filme D |
| --------------- | ------: | ------: | ------: | ------: |
| Usuário 1       |       5 |       ? |       3 |       ? |
| Usuário 2       |       ? |       4 |       ? |       2 |
| Usuário 3       |       4 |       ? |       ? |       5 |

O símbolo `?` significa que o usuário não avaliou aquele filme.

O problema é justamente tentar prever esses valores faltantes.

---

## 4. Modelos que serão comparados

## 4.1 Modelo 1: Popularidade

Este é o modelo mais simples.

Ele recomenda os filmes com maior média de avaliação ou com maior quantidade de avaliações.

Exemplo:

Se muitos usuários avaliaram bem o filme `Toy Story`, então ele será recomendado para vários usuários.

Esse método não é personalizado, pois recomenda quase os mesmos filmes para todo mundo.

### Fórmula usada

A nota prevista para um filme pode ser a média das notas que ele recebeu:

[
score(i) = \frac{1}{N_i} \sum_{u \in U_i} r_{ui}
]

Onde:

* (i) é o filme;
* (u) é o usuário;
* (r_{ui}) é a nota que o usuário (u) deu para o filme (i);
* (N_i) é o número de avaliações do filme (i);
* (U_i) é o conjunto de usuários que avaliaram o filme (i).

### Como ele recomenda

Para cada usuário:

1. Verifica quais filmes ele ainda não assistiu ou avaliou;
2. Ordena esses filmes pela média de avaliação;
3. Recomenda os melhores.

---

## 4.2 Modelo 2: SVD

O SVD é uma técnica de Álgebra Linear que decompõe uma matriz em outras três matrizes.

A ideia é aproximar a matriz original de avaliações por uma versão menor.

### Modelo matemático

A matriz de avaliações será chamada de:

[
R
]

O SVD tenta decompor essa matriz assim:

[
R \approx U \Sigma V^T
]

Onde:

* (R) é a matriz usuário-filme;
* (U) representa informações dos usuários;
* (\Sigma) representa a importância de cada fator;
* (V^T) representa informações dos filmes.

Na prática, vamos usar apenas os primeiros (k) fatores:

[
R \approx U_k \Sigma_k V_k^T
]

Isso é chamado de **SVD truncado**.

### O que significa o valor (k)?

O valor (k) representa quantas características escondidas o modelo vai usar.

Por exemplo:

```text
k = 10
k = 20
k = 50
k = 100
```

Essas características não têm nome, mas podem representar padrões como:

* usuários que gostam de filmes de ação;
* usuários que gostam de romance;
* filmes antigos;
* filmes populares;
* filmes de comédia.

O modelo aprende esses padrões sozinho a partir dos números.

### Problema do SVD

O SVD tradicional precisa de uma matriz completa.

Mas a matriz de avaliações tem muitos valores faltando.

Por isso, será necessário preencher os valores faltantes antes de aplicar o SVD.

Uma estratégia simples será:

1. Subtrair a média de notas de cada usuário;
2. Colocar zero nos valores faltantes;
3. Aplicar o SVD;
4. Reconstruir a matriz;
5. Somar a média do usuário novamente.

### Fórmula da reconstrução

Depois do SVD, a nota prevista será:

[
\hat{R} = U_k \Sigma_k V_k^T
]

Onde:

* (\hat{R}) é a matriz reconstruída;
* cada valor (\hat{r}_{ui}) representa a nota prevista para o usuário (u) no filme (i).

---

## 4.3 Modelo 3: Matrix Factorization com embeddings

Este será o modelo principal do projeto.

Apesar do SVD também ser uma fatoração de matriz, neste projeto o nome **Matrix Factorization** será usado para o modelo treinado com embeddings.

A ideia é representar cada usuário e cada filme por uma lista pequena de números.

Exemplo:

```text
Usuário 1 = [0.8, 0.2, 0.5]
Filme A   = [0.7, 0.1, 0.6]
```

Essas listas são chamadas de **embeddings**.

Se o vetor do usuário combina bem com o vetor do filme, o sistema prevê que o usuário vai gostar daquele filme.

---

## 5. Modelo matemático da Matrix Factorization

A previsão da nota será feita assim:

[
\hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i
]

Onde:

* (\hat{r}_{ui}) é a nota prevista;
* (\mu) é a média geral das notas;
* (b_u) é o ajuste do usuário;
* (b_i) é o ajuste do filme;
* (p_u) é o embedding do usuário;
* (q_i) é o embedding do filme;
* (p_u^T q_i) é o produto entre os dois vetores.

---

## 6. Explicação simples da fórmula

A nota prevista tem quatro partes:

### 1. Média geral

[
\mu
]

Exemplo:

Se a média geral das avaliações for 3.5, começamos a previsão por 3.5.

### 2. Ajuste do usuário

[
b_u
]

Alguns usuários dão notas mais altas que outros.

Por exemplo, um usuário pode ter costume de dar muitas notas 5.

Outro pode ser mais crítico e dar notas menores.

### 3. Ajuste do filme

[
b_i
]

Alguns filmes são naturalmente mais bem avaliados.

Por exemplo, um filme muito famoso pode ter média alta.

### 4. Combinação usuário-filme

[
p_u^T q_i
]

Essa parte mede o quanto o gosto do usuário combina com as características do filme.

---

## 7. Função de erro

O modelo precisa aprender os melhores embeddings.

Para isso, ele tenta diminuir o erro entre a nota real e a nota prevista.

A função de erro será:

[
L = \sum_{(u,i) \in K} (r_{ui} - \hat{r}_{ui})^2 + \lambda(|p_u|^2 + |q_i|^2 + b_u^2 + b_i^2)
]

Onde:

* (L) é o erro total;
* (K) é o conjunto de avaliações conhecidas;
* (r_{ui}) é a nota real;
* (\hat{r}_{ui}) é a nota prevista;
* (\lambda) controla a regularização.

A regularização serve para evitar que o modelo decore os dados de treino e vá mal nos dados de teste.

---

## 8. Como o modelo aprende

O modelo será treinado usando **gradiente descendente estocástico**, também chamado de SGD.

A ideia é:

1. Começar com embeddings aleatórios;
2. Escolher uma avaliação conhecida;
3. Prever a nota;
4. Calcular o erro;
5. Ajustar os embeddings;
6. Repetir isso várias vezes.

### Erro de uma previsão

[
e_{ui} = r_{ui} - \hat{r}_{ui}
]

### Atualização dos parâmetros

[
b_u \leftarrow b_u + \eta(e_{ui} - \lambda b_u)
]

[
b_i \leftarrow b_i + \eta(e_{ui} - \lambda b_i)
]

[
p_u \leftarrow p_u + \eta(e_{ui}q_i - \lambda p_u)
]

[
q_i \leftarrow q_i + \eta(e_{ui}p_u - \lambda q_i)
]

Onde:

* (\eta) é a taxa de aprendizado;
* (\lambda) é a regularização;
* (e_{ui}) é o erro da previsão.

---

## 9. Separação dos dados

Os dados serão divididos em:

* **treino**
* **teste**

Uma divisão possível:

```text
80% das avaliações para treino
20% das avaliações para teste
```

O modelo só poderá aprender usando os dados de treino.

Os dados de teste ficam escondidos para verificar se o modelo realmente consegue prever avaliações novas.

---

## 10. Avaliação dos modelos

Os modelos serão avaliados de duas formas:

1. Erro na previsão das notas;
2. Qualidade das recomendações Top-N.

---

## 10.1 RMSE

O RMSE mede o erro médio das previsões.

[
RMSE = \sqrt{\frac{1}{|T|} \sum_{(u,i) \in T} (r_{ui} - \hat{r}_{ui})^2}
]

Onde:

* (T) é o conjunto de teste;
* (r_{ui}) é a nota real;
* (\hat{r}_{ui}) é a nota prevista.

Quanto menor o RMSE, melhor.

---

## 10.2 MAE

O MAE também mede erro de previsão, mas de forma mais simples.

[
MAE = \frac{1}{|T|} \sum_{(u,i) \in T} |r_{ui} - \hat{r}_{ui}|
]

Quanto menor o MAE, melhor.

---

## 10.3 Precision@K

O Precision@K mede quantos filmes recomendados realmente eram relevantes.

Neste projeto, um filme será considerado relevante se o usuário deu nota maior ou igual a 4.

[
Precision@K = \frac{\text{número de filmes relevantes recomendados}}{K}
]

Exemplo:

Se o sistema recomenda 10 filmes e 3 deles eram realmente relevantes:

[
Precision@10 = \frac{3}{10} = 0.3
]

---

## 10.4 Recall@K

O Recall@K mede quantos dos filmes relevantes o sistema conseguiu encontrar.

[
Recall@K = \frac{\text{número de filmes relevantes recomendados}}{\text{número total de filmes relevantes no teste}}
]

Exemplo:

Se o usuário tinha 5 filmes relevantes no teste e o sistema recomendou 2 deles:

[
Recall@10 = \frac{2}{5} = 0.4
]

---

## 11. Experimentos planejados

## Experimento 1: Comparar o erro dos modelos

Neste experimento, vamos comparar o RMSE e o MAE dos três modelos:

| Modelo               |        RMSE |         MAE |
| -------------------- | ----------: | ----------: |
| Popularidade         | A preencher | A preencher |
| SVD                  | A preencher | A preencher |
| Matrix Factorization | A preencher | A preencher |

Resultado esperado:

```text
Matrix Factorization deve ter o menor erro.
SVD deve ser melhor que popularidade.
Popularidade deve ser o modelo mais simples e menos personalizado.
```

### Imagem do experimento

Inserir aqui o gráfico comparando RMSE dos modelos.

```markdown
![Comparação de RMSE](docs/images/rmse_comparacao.png)
```

---

## Experimento 2: Comparar recomendações Top-10

Neste experimento, vamos comparar a qualidade das recomendações dos modelos.

| Modelo               | Precision@10 |   Recall@10 |
| -------------------- | -----------: | ----------: |
| Popularidade         |  A preencher | A preencher |
| SVD                  |  A preencher | A preencher |
| Matrix Factorization |  A preencher | A preencher |

Resultado esperado:

```text
Matrix Factorization deve recomendar filmes mais personalizados.
Popularidade pode recomendar filmes bons, mas iguais para muitos usuários.
SVD deve ficar no meio.
```

### Imagem do experimento

Inserir aqui o gráfico comparando Precision@10 e Recall@10.

```markdown
![Comparação Top-10](docs/images/top10_comparacao.png)
```

---

## Experimento 3: Testar diferentes valores de k

Neste experimento, vamos mudar a quantidade de fatores latentes.

Valores planejados:

```text
k = 5
k = 10
k = 20
k = 50
k = 100
```

O objetivo é verificar se aumentar o número de fatores melhora ou piora o resultado.

|   k |    RMSE SVD | RMSE Matrix Factorization |
| --: | ----------: | ------------------------: |
|   5 | A preencher |               A preencher |
|  10 | A preencher |               A preencher |
|  20 | A preencher |               A preencher |
|  50 | A preencher |               A preencher |
| 100 | A preencher |               A preencher |

Resultado esperado:

```text
Poucos fatores podem ser insuficientes.
Muitos fatores podem causar overfitting.
Um valor intermediário deve funcionar melhor.
```

### Imagem do experimento

Inserir aqui o gráfico mostrando o efeito do valor de k.

```markdown
![Efeito do valor de k](docs/images/efeito_k.png)
```

---

## Experimento 4: Exemplo de recomendação para um usuário

Neste experimento, vamos escolher alguns usuários e mostrar os filmes recomendados para eles.

Exemplo:

### Usuário escolhido

```text
userId = A preencher
```

### Filmes que o usuário já avaliou bem

| Filme       |        Nota |
| ----------- | ----------: |
| A preencher | A preencher |
| A preencher | A preencher |
| A preencher | A preencher |

### Recomendações do modelo de popularidade

| Posição | Filme recomendado |
| ------: | ----------------- |
|       1 | A preencher       |
|       2 | A preencher       |
|       3 | A preencher       |
|       4 | A preencher       |
|       5 | A preencher       |

### Recomendações do SVD

| Posição | Filme recomendado |
| ------: | ----------------- |
|       1 | A preencher       |
|       2 | A preencher       |
|       3 | A preencher       |
|       4 | A preencher       |
|       5 | A preencher       |

### Recomendações da Matrix Factorization

| Posição | Filme recomendado |
| ------: | ----------------- |
|       1 | A preencher       |
|       2 | A preencher       |
|       3 | A preencher       |
|       4 | A preencher       |
|       5 | A preencher       |

### Imagem do experimento

Inserir aqui uma imagem ou tabela final das recomendações.

```markdown
![Exemplo de recomendação](docs/images/exemplo_recomendacao.png)
```

---

## 12. Estrutura esperada do projeto

```text
projeto-recomendacao-filmes/
│
├── data/
│   ├── ratings.csv
│   ├── movies.csv
│
├── notebooks/
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_popularidade.ipynb
│   ├── 03_svd.ipynb
│   ├── 04_matrix_factorization.ipynb
│   ├── 05_avaliacao.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── popularity_model.py
│   ├── svd_model.py
│   ├── mf_model.py
│   ├── metrics.py
│
├── docs/
│   ├── images/
│   │   ├── matriz_esparsa.png
│   │   ├── rmse_comparacao.png
│   │   ├── top10_comparacao.png
│   │   ├── efeito_k.png
│   │   ├── exemplo_recomendacao.png
│
├── README.md
└── requirements.txt
```

---

## 13. Etapas de desenvolvimento

## Etapa 1: Carregar os dados

Nesta etapa, os arquivos `ratings.csv` e `movies.csv` serão carregados usando Python.

Também será feita uma análise simples:

* quantidade de usuários;
* quantidade de filmes;
* quantidade de avaliações;
* média das notas;
* distribuição das notas.

### Imagem esperada

Inserir aqui gráfico da distribuição das notas.

```markdown
![Distribuição das notas](docs/images/distribuicao_notas.png)
```

---

## Etapa 2: Criar a matriz usuário-filme

Será criada uma matriz onde:

* linhas são usuários;
* colunas são filmes;
* valores são notas.

Como nem todo usuário avaliou todo filme, essa matriz será esparsa, ou seja, terá muitos valores vazios.

### Imagem esperada

Inserir aqui imagem mostrando a matriz esparsa.

```markdown
![Matriz usuário-filme](docs/images/matriz_esparsa.png)
```

---

## Etapa 3: Separar treino e teste

As avaliações serão divididas em treino e teste.

O treino será usado para os modelos aprenderem.

O teste será usado apenas para avaliação.

Importante:

```text
O modelo não pode ver os dados de teste durante o treino.
```

---

## Etapa 4: Implementar o modelo de popularidade

O modelo de popularidade será usado como comparação inicial.

Ele é importante porque mostra se os modelos mais avançados realmente estão melhorando as recomendações.

---

## Etapa 5: Implementar o SVD

O SVD será aplicado na matriz usuário-filme.

Será necessário tratar os valores faltantes antes da decomposição.

Depois, a matriz será reconstruída para gerar previsões de notas.

---

## Etapa 6: Implementar Matrix Factorization

Nesta etapa, serão criados embeddings para usuários e filmes.

O modelo será treinado usando as avaliações conhecidas.

A cada época de treino, o erro deve diminuir.

### Imagem esperada

Inserir aqui gráfico do erro durante o treino.

```markdown
![Erro durante o treino](docs/images/erro_treino_mf.png)
```

---

## Etapa 7: Avaliar os modelos

Os três modelos serão comparados usando:

* RMSE;
* MAE;
* Precision@10;
* Recall@10.

---

## Etapa 8: Interpretar os resultados

Depois dos testes, será feita uma análise respondendo:

1. Qual modelo teve menor erro?
2. Qual modelo recomendou melhor?
3. O modelo de Matrix Factorization superou o SVD?
4. O modelo de popularidade foi suficiente?
5. Quais limitações apareceram?

---

## 14. Resultados esperados

A expectativa inicial é:

| Modelo               | Resultado esperado                                          |
| -------------------- | ----------------------------------------------------------- |
| Popularidade         | Simples, mas pouco personalizado                            |
| SVD                  | Melhor que popularidade, mas limitado por valores faltantes |
| Matrix Factorization | Melhor desempenho geral                                     |

---

## 15. Limitações do projeto

Algumas limitações esperadas:

1. O dataset é pequeno em comparação com sistemas reais.
2. O modelo usa apenas notas, sem usar descrição, elenco ou gênero dos filmes.
3. Usuários com poucas avaliações são mais difíceis de recomendar.
4. Filmes com poucas avaliações também são difíceis de recomendar.
5. O SVD precisa lidar com valores faltantes de forma artificial.
6. Matrix Factorization pode sofrer overfitting se usar muitos fatores.

---

## 16. Possíveis melhorias futuras

Algumas melhorias que poderiam ser feitas depois:

1. Usar gêneros dos filmes.
2. Usar ano de lançamento.
3. Usar tags dos usuários.
4. Testar outros modelos, como KNN ou redes neurais.
5. Fazer recomendações apenas com feedback implícito.
6. Comparar com modelos baseados em grafos.

---

## 17. Conclusão esperada

Este projeto mostra como um problema real de recomendação pode ser modelado usando Álgebra Linear.

A matriz usuário-filme representa as avaliações conhecidas.

O SVD aplica uma decomposição clássica de matrizes.

A Matrix Factorization aprende embeddings de usuários e filmes para prever avaliações faltantes.

No final, os modelos serão comparados para verificar qual deles consegue recomendar filmes de forma mais eficiente e personalizada.

---

## 18. Como executar

Instalar as dependências:

```bash
pip install -r requirements.txt
```

Executar os notebooks na seguinte ordem:

```text
01_exploracao_dados.ipynb
02_popularidade.ipynb
03_svd.ipynb
04_matrix_factorization.ipynb
05_avaliacao.ipynb
```

Ou executar os scripts em `src/`, caso o projeto seja organizado em arquivos `.py`.

---

## 19. Dependências planejadas

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

---

## 20. Status do projeto

* [ ] Baixar dataset
* [ ] Carregar dados
* [ ] Explorar dados
* [ ] Criar matriz usuário-filme
* [ ] Separar treino e teste
* [ ] Implementar popularidade
* [ ] Implementar SVD
* [ ] Implementar Matrix Factorization
* [ ] Avaliar RMSE e MAE
* [ ] Avaliar Precision@10 e Recall@10
* [ ] Gerar gráficos
* [ ] Escrever relatório final


