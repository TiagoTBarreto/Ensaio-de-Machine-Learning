# 1. Ensaio de Machine Learning

# 2. Problema de Negócio
## 2.1 Descrição
A empresa Data Money acredita que a expertise no treinamento e ajuste fino dos algoritmos, feito pelos Cientistas de Dados da empresa, é a principal motivo dos ótimos resultados que as consultorias vem entregando aos seus clientes.
## 2.2 Objetivo
O objetivo desse projeto será realizar ensaios com algoritmos de Classificação, Regressão e Clusterização, para estudar a mudança do comportamento da performance, a medida que os valores dos principais parâmetros de controle de overfitting e underfitting mudam.

# 3. Planejamento da solução
## 3.1 Produto final
O produto final será 7 tabelas mostrando a performance dos algoritmos, avaliados usando múltiplas
métricas, para 3 conjuntos de dados diferentes: Treinamento, validação e teste.

## 3.2 Algoritmos ensaiados

### Classificação:
  1. Algoritmos: KNN, Decision Tree, Random Forest e Logistic Regression
  2. Métricas de performance: Accuracy, Precision, Recall e F1-Score

### Regressão:
  1. Algoritmos: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Polinomial Regression, Linear Regression Lasso, Linear Regression Ridge, Linear Regression Elastic Net, Polinomial Regression Lasso, Polinomial Regression Ridge e     Polinomial Regression Elastic Net
  2. Métricas de performance: R2, MSE, RMSE, MAE e MAPE

### Agrupamento:
  1. Algoritmos: K-Means e Affinity Propagation
  2. Métricas de performance: Silhouette Score

## 3.3 Ferramentas utilizadas:
  1. Python 3.8 e Scikit-learn

# 4. Desenvolvimento
## 4.1 Estratégia da solução
Para o objetivo de ensaiar os algoritmos de Machine Learning, eu vou escrever os códigos utilizando a linguagem Python, para treinar cada um dos algoritmos e vou variar seus principais parâmetros de ajuste de overfitting e observar a métrica final.

O conjunto de valores que fizerem os algoritmos alcançarem a melhor performance, serão aqueles escolhidos para o treinamento final do algoritmo.

## 4.2 O passo a passo
  1. Divisão dos dados em treino, teste e validação.
  2. Treinamento dos algoritmos com os dados de treinamento, utilizando os parâmetros “default”.
  3. Medir a performance dos algoritmos treinados com o parâmetro default, utilizando o conjunto de dados de treinamento.
  4. Medir a performance dos algoritmos treinados com o parâmetro “default”, utilizando o conjunto de dados de validação.
  5. Alternar os valores dos principais parâmetros que controlam o overfitting do algoritmo até encontrar o conjunto de parâmetros apresente a melhor performance dos algoritmos.
  6. Unir os dados de treinamento e validação
  7. Retreinar o algoritmo com a união dos dados de treinamento e validação, utilizando os melhores valores para os parâmetros de controle do algoritmo.
  8. Medir a performance dos algoritmos treinados com os melhores parâmetro, utilizando o conjunto de dados de teste.
  9. Avaliar os ensaios e anotar os 3 principais Insights que se destacaram.

# 5. Os top 3 Insights

## 5.1 Insight Top 1
Em relação ao ensaio de classificação, os modelos baseados em árvores obtiveram as melhores métricas de performance e mesmo mudando de dataset entre treinamento, validação e teste houve uma pequena diminuição das métricas. Concluindo assim que o algoritmo não teve overfitting. 
## 5.2 Insight Top 2
Em relação ao ensaio de regressão nenhum algoritmo obteve uma performance muito boa, então para o algoritmo conseguir realizar a tarefa de regressão com precisão seria necessário realizar uma seleção de features, criação de features com base nas já existentes e a coleta de mais dados.
## 5.3 Insight Top 3
Durante o ensaio de regressão, o algoritmo Decision Tree Regressor chegou a quase um R² de 1.00 demonstrando uma grande representatividade dos dados, mas essa grande representativade foi devido ao algoritmo ter decorado os dados, isso pode ser confirmado quando ele foi submetido a dados que o algoritmo nunca tinha visto, chegando a um R² negativo. 

# 6. Resultados
## Ensaio de Classificação:
### 6.1 Sobre os dados de treinamento
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/9b6f9f9a-fbff-4923-b3f0-99459148dd9e)
### 6.2 Sobre os dados de validação
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/9b07499c-da26-4018-8833-ed541648dafd)
### 6.3 Sobre os dados de teste
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/a3572d32-de15-4af6-9b8d-e18e3b48840d)

## Ensaio de Regressão:
### 6.4 Sobre os dados de treinamento 
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/ce44308e-b9bc-4ad6-9b17-6a7db40bc2d7)
### 6.5 Sobre os dados de validação
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/1d9e353b-7b56-47a7-96de-7dc9fc6d2325)
### 6.6 Sobre os dados de teste
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/83411a8e-c8bc-4912-9a96-0c81d44a489b)

## Ensaio de Clusterização:
### 6.7 Sobre os dados de treinamento
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/d6e7c618-3678-4e19-bc16-07b03162ec85)


# 8. Próximos passos
Como próximos passos desse ensaio, pretendo ensaiar novos algoritmos de Machine Learning aumentando assim o meu leque de ferramentas para a resolução de problemas. Além de testar esses modelos em novos datasets para conseguir ir aprimorando cada vez mais minha visão de quais algoritmos e quais parâmetros performam melhor de acordo com a particularidade de cada conjunto de dados.






