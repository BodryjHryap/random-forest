from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def check_n_trees(n_trees):
  if n_trees > 100 or n_trees < 1:
    return False
  return True

def check_depth(depth):
  if depth > 20 or depth < 1:
    return False
  return True

def learn_model(X_train, y_train, n_trees, depth):
  rf = RandomForestClassifier(n_estimators=n_trees, max_depth=depth)
  rfc=rf.fit(X_train, y_train)
  return rfc

def classify(model, X_test, y_test):
  y_pred = model.predict(X_test)
  acc = accuracy_score(y_pred,y_test)
  return acc