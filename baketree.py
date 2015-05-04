__author__ = 'evanzamir'

import json

from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals.six import StringIO
import pydot

ingredients = []
items = []

with open('recipes.json') as f:
  recipes = json.load(f)
  for recipe in recipes:
    ingredients.append(recipe['ingredients'])
    items.append(recipe['name'])

v = DictVectorizer(sparse=False)
X = v.fit_transform(ingredients)
clf = tree.DecisionTreeClassifier()
model = clf.fit(X, items)


def main():
  print(X)
  print(items)
  print(clf)
  with open('bake.dot', 'w') as f:
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data)
  graph = pydot.graph_from_dot_data(dot_data.getvalue())
  graph.write_pdf('bake.pdf')
  print(clf.classes_)


if __name__ == "__main__":
  main()
