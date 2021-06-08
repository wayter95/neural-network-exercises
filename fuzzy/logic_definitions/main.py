import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# DEFINIÇÕES DOS UNIVERSOS

# antecedentes, que serão as predições indicando a probabilidade do imóvel ser vendido e o seu valor
venda = ctrl.Antecedent(np.arange(0, 1.01, .01), 'venda')
preco = ctrl.Antecedent(np.arange(100000, 1001000, 1000), 'preco')

# consequente, que será a qualidade do imóvel
qualidade = ctrl.Consequent(np.arange(0, 100.1, .1), 'qualidade')

# DEFINIÇÕES DAS FUNÇÕES DE PERTINÊNCIA
# venda
venda['baixa'] = fuzz.trimf(venda.universe, [0, 0.25, 0.5])
venda['media'] = fuzz.trimf(venda.universe, [0.3,0.5,0.7])
venda['alta'] = fuzz.trimf(venda.universe, [0.6,0.8,1])

# preço
preco['baixo'] = fuzz.trimf(preco.universe, [100000, 200000,300000])
preco['medio'] = fuzz.trimf(preco.universe, [250000, 500000, 700000])
preco['alto'] = fuzz.trimf(preco.universe, [600000, 800000, 1000000])

# qualidade
qualidade['ruim'] = fuzz.trapmf(qualidade.universe, [0, 25, 45,60])
qualidade['mediana'] = fuzz.trapmf(qualidade.universe, [40, 50, 70,80])
qualidade['boa'] = fuzz.trapmf(qualidade.universe, [75, 80,95,100])

# VISUALIZAÇÃO DAS FUNÇÕES DE PERTINÊNCIA
# venda
venda.view()

# preco
preco.view()

# qualidade
qualidade.view()

# regra 1 - se probabilidade venda é baixa, então qualidade é ruim
regra1 = ctrl.Rule(venda['baixa'], qualidade['ruim'])

# regra 2 - se probabilidade venda é médio ou o preço é médio, então qualidade é mediana
regra2 = ctrl.Rule(venda['media'] | preco['medio'], qualidade['mediana'])

# regra 3 - se probabilidade venda é alta e o preço é alto, então qualidade é boa
regra3 = ctrl.Rule(venda['alta'] & preco['alto'], qualidade['boa'])

# regra 4 - se probabilidade venda é médio ou o preço é baixo, então qualidade é mediana
regra4 = ctrl.Rule(venda['media'] | preco['baixo'], qualidade['mediana'])

# regra 5 - se probabilidade venda é baixa e o preço é alto, então qualidade é mediana
regra5 = ctrl.Rule(venda['baixa'] & preco['alto'], qualidade['mediana'])

imovel_ctrl = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5])
engine = ctrl.ControlSystemSimulation(imovel_ctrl)

# passa as predições dos modelos para suas respectivas variáveis de entrada
engine.input['venda'] = 0.8
engine.input['preco'] = 150000

# calcula a saída do sistema de controle fuzzy
engine.compute()

# retorna o valor crisp e o gráfico mostrando-o
print(engine.output['qualidade'])
qualidade.view(sim=engine)