# Nome do Projeto
### Ensaio de Machine Learning

# Problema de Negócio
## Descrição
A empresa Data Money acredita que a expertise no treinamento e ajuste fino dos algoritmos, feito pelos Cientistas de Dados da empresa, é a principal motivo dos ótimos resultados que as consultorias vem entregando aos seus clientes.
## Objetivo
O objetivo desse projeto será realizar ensaios com algoritmos de Classificação, Regressão e Clusterização, para estudar a mudança do comportamento da performance, a medida que os valores dos principais parâmetros de controle de overfitting e underfitting mudam.

# Planejamento da solução
## Produto final
O produto final será 7 tabelas mostrando a performance dos algoritmos, avaliados usando múltiplas
métricas, para 3 conjuntos de dados diferentes: Treinamento, validação e teste.

## Algoritmos ensaiados

### Classificação:
  1. Algoritmos: KNN, Decision Tree, Random Forest e Logistic Regression
  2. Métricas de performance: Accuracy, Precision, Recall e F1-Score

### Regressão:
  1. Algoritmos: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Polinomial Regression, Linear Regression Lasso, Linear Regression Ridge, Linear Regression Elastic Net, Polinomial Regression Lasso, Polinomial Regression Ridge e     Polinomial Regression Elastic Net
  2. Métricas de performance: R2, MSE, RMSE, MAE e MAPE

### Agrupamento:
  1. Algoritmos: K-Means e Affinity Propagation
  2. Métricas de performance: Silhouette Score

## Ferramentas utilizadas:
  1. Python 3.8 e Scikit-learn

# Desenvolvimento
## Estratégia da solução
Para o objetivo de ensaiar os algoritmos de Machine Learning, eu vou escrever os códigos utilizando a linguagem Python, para treinar cada um dos algoritmos e vou variar seus principais parâmetros de ajuste de overfitting e observar a métrica final.

O conjunto de valores que fizerem os algoritmos alcançarem a melhor performance, serão aqueles escolhidos para o treinamento final do algoritmo.

## O passo a passo
  1. Divisão dos dados em treino, teste e validação.
  2. Treinamento dos algoritmos com os dados de treinamento, utilizando os parâmetros “default”.
  3. Medir a performance dos algoritmos treinados com o parâmetro default, utilizando o conjunto de dados de treinamento.
  4. Medir a performance dos algoritmos treinados com o parâmetro “default”, utilizando o conjunto de dados de validação.
  5. Alternar os valores dos principais parâmetros que controlam o overfitting do algoritmo até encontrar o conjunto de parâmetros apresente a melhor performance dos algoritmos.
  6. Unir os dados de treinamento e validação
  7. Retreinar o algoritmo com a união dos dados de treinamento e validação, utilizando os melhores valores para os parâmetros de controle do algoritmo.
  8. Medir a performance dos algoritmos treinados com os melhores parâmetro, utilizando o conjunto de dados de teste.
  9. Avaliar os ensaios e anotar os 3 principais Insights que se destacaram.

# Os top 3 Insights

## Insight Top 1
Durante o ensaio de classificação, os modelos baseados em árvores obtiveram as melhores métricas de performance e mesmo submetendo o algoritmo a dados não vistos no treinamento houve uma pequena diminuição das métricas. Concluindo assim que o algoritmo não teve overfitting. 
## Insight Top 2
Durante o ensaio de regressão nenhum algoritmo obteve uma performance muito boa, então para o algoritmo conseguir realizar a tarefa de regressão com precisão seria necessário realizar uma seleção de features, criação de features com base nas já existentes e a coleta de mais dados.
## Insight Top 3
Durante o ensaio de regressão, o algoritmo Decision Tree Regressor chegou a quase um R² de 1.00 demonstrando que o modelo representava quase em sua totalidade a variação dos dados, mas essa grande representativade foi devido ao algoritmo ter decorado os dados de treinemanto, isso pode ser confirmado quando ele foi submetido a novos dados e que o algoritmo nunca tinha visto, chegando a um R² negativo. 

# Resultados
## Ensaio de Classificação:
### Sobre os dados de treinamento
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/9b6f9f9a-fbff-4923-b3f0-99459148dd9e)
### Sobre os dados de validação
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/9b07499c-da26-4018-8833-ed541648dafd)
### Sobre os dados de teste
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/a3572d32-de15-4af6-9b8d-e18e3b48840d)

## Ensaio de Regressão:
### Sobre os dados de treinamento 
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/ce44308e-b9bc-4ad6-9b17-6a7db40bc2d7)
### Sobre os dados de validação
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/1d9e353b-7b56-47a7-96de-7dc9fc6d2325)
### Sobre os dados de teste
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/83411a8e-c8bc-4912-9a96-0c81d44a489b)

## Ensaio de Clusterização:
### Sobre os dados de treinamento
![image](https://github.com/TiagoTBarreto/Ensaio-de-Machine-Learning/assets/137197787/d6e7c618-3678-4e19-bc16-07b03162ec85)

# Conclusões

Nesse ensaio de Machine Learning, consegui adquirir experiência e entender melhor sobre os limites dos algoritmos entre os estados de underffiting e overfitting.

Os algoritmos baseados em árvores são sensíveis aos números de árvores e a profundidade das árvores, já que aumentando a profundidade da árvore acaba adaptando cada vez mais o algoritmo aos dados de treinamento levando ao overfitting.

Por outro lado os algoritmos de regressão polinomial, o principal parâmetro que leva ao overfitting é o grau do polinômio. Durante o ensaio quanto mais aumentava o grau, melhor o algoritmo performava nos dados de treino mas ao realizar a validação e o teste ele acaba tendo um desempenho péssimo, pois o grau do polinômio faz com que o modelo se adapte aos dados de treino levando assim ao overfitting. Ainda em relação aos algoritmos de regressão, foi possível observar que nesse dataset a regularização Lasso e Elastic não tiveram um desempenho nem parecido com o modelo sem regularização, já a regularização Ridge performou parecido com os algoritmos normais mas mesmo assim não chegaram nem perto de atender as métricas de performance necessárias para colocar os modelos em produção.

Esse ensaio de Machine Learning foi muito importante para aprofundar o entendimento sobre o funcionamento de diversos algoritmos de classificação, regressão e clusterização e quais os principais parâmetros de controle entre os estados de underfitting e overfitting.

# Próximos passos
Como próximos passos desse ensaio, pretendo ensaiar novos algoritmos de Machine Learning aumentando assim o meu leque de ferramentas para a resolução de problemas. Além de testar esses modelos em novos datasets para conseguir ir aprimorando cada vez mais minha visão de quais algoritmos e quais parâmetros performam melhor de acordo com a particularidade de cada conjunto de dados.






