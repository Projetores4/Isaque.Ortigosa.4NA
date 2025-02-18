# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Gerando dados de exemplo
np.random.seed(42)  # Para reprodutibilidade
x = np.linspace(0, 10, 100)
# Criando uma relação linear com algum ruído aleatório
y = 2.5 + 1.3 * x + np.random.normal(0, 1, 100)

# Organizando os dados em um DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Adicionando uma constante à variável independente (necessário para o intercepto)
X = sm.add_constant(df['x'])

# Ajustando o modelo de regressão linear usando o método OLS (Ordinary Least Squares)
modelo = sm.OLS(df['y'], X).fit()

# Exibindo o resumo dos resultados do modelo
print(modelo.summary())

# Plotando os dados e a linha de regressão
plt.scatter(x, y, label='Dados Observados')
plt.plot(x, modelo.predict(X), color='red', label='Linha de Regressão')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regressão Linear Simples')
plt.legend()
plt.show()
