__author__ = 'evanzamir'

import json
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals.six import StringIO
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import pydot

ingredients = []
items = []

with open('recipes.json') as f:
  recipes = json.load(f)
  for recipe in recipes:
    ingredients.append(recipe['ingredients'])
    items.append(recipe['name'])


def main():
  v = DictVectorizer(sparse=False)
  X = v.fit_transform(ingredients)
  features = [str(x) for x in v.get_feature_names()]
  print(features)
  clf = tree.DecisionTreeClassifier(criterion='gini')
  model = clf.fit(X, items)
  ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X)
  plt.figure()
  # fig = plt.figure()
  print(ward.labels_)

  print(X)
  print(items)
  print(clf)
  with open('bake.dot', 'w') as f:
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data, feature_names=features)
  graph = pydot.graph_from_dot_data(dot_data.getvalue())
  graph.write_pdf('bake.pdf')
  print(clf.classes_)
  for row, item in enumerate(items):
    plt.scatter(X_pca[row, 0], X_pca[row, 1], s=100, c='rgbykc'[ward.labels_[row]])
    plt.annotate("{}:{}".format(item, ward.labels_[row]),
                 xy=(X_pca[row, 0], X_pca[row, 1]))
  plt.show()


if __name__ == "__main__":
  main()
