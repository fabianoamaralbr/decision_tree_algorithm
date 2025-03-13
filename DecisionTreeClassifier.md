# Árvore de Classificação

| Atributo | Descrição |
| --- | --- |
| `criterion` | O critério de função para medir a qualidade de uma divisão. Pode ser "gini" para o índice de Gini ou "entropy" para a informação de entropia. |
| `splitter` | A estratégia usada para escolher a divisão em cada nó. Pode ser "best" para escolher a melhor divisão ou "random" para escolher a melhor divisão aleatória. |
| `max_depth` | A profundidade máxima da árvore. Se None, os nós são expandidos até que todas as folhas sejam puras ou até que todas as folhas contenham menos do que min_samples_split amostras. |
| `min_samples_split` | O número mínimo de amostras necessárias para dividir um nó interno. |
| `min_samples_leaf` | O número mínimo de amostras necessárias para estar em um nó folha. |
| `min_weight_fraction_leaf` | A fração mínima de peso das amostras necessárias para estar em um nó folha. |
| `max_features` | O número de características a serem consideradas ao procurar a melhor divisão. |
| `random_state` | Controla a aleatoriedade do estimador. |
| `max_leaf_nodes` | Cresce uma árvore com no máximo max_leaf_nodes em melhores splits. |
| `min_impurity_decrease` | Um nó será dividido se essa divisão induzir uma diminuição da impureza maior ou igual a este valor. |
| `class_weight` | Pesos associados às classes. Se None, todos os pesos das classes são considerados iguais. |
| `ccp_alpha` | Complexidade de parâmetro de poda mínima. Utilizado para a poda mínima de custo-complexidade. |

- **Exemplo de uso**

```python
from sklearn.tree import DecisionTreeClassifier

# Criação do modelo
clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1)

# Treinamento do modelo
clf.fit(X_train, y_train)

# Predição
y_pred = clf.predict(X_test)
```