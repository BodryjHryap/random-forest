import components
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('random-forest/dataset/Training_Dataset_v2.csv')
X = data.drop(labels='went_on_backorder',axis=1)
y = data['went_on_backorder']
X_train,X_test,y_train,y_test = train_test_split(X,y)
while True:
  print('Программа для предсказания повторного заказа товара. Выберите дальнейшее действие:\n\t1 - Обучение и классификация\n\t2 - Выход')
  action = input('>')
  if action == '1':
    n_trees = int(input('Введите количество деревьев (1-100): '))
    if not components.check_n_trees(n_trees):
      print('Некорректное число. Значение установлено по-умолчанию.')
      n_trees = 20
    depth = int(input('Введите глубину дерева (1-20): '))
    if not components.check_depth(depth):
      print('Некорректное число. Значение установлено по-умолчанию.')
      depth = None
    model = components.learn_model(X_train, y_train, n_trees, depth)
    accuracy = components.classify(model, X_test, y_test)
    print(accuracy)
  elif action == '2':
    break
  else:
    print('Ошибка, повторите попытку!')