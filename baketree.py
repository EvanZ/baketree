__author__ = 'evanzamir'

import json
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, TransformerMixin, BaseEstimator


ingredients = []
items = []


class AgglomerativeWrapper(BaseEstimator, TransformerMixin):
  def __init__(self, model):
    self.model = model

  def fit(self, x, y=None):
    self.labels_ = self.model.fit_predict(x)
    return self


def main():
  with open('recipes.json') as f:
    recipes = json.load(f)
    for recipe in recipes:
      ingredients.append(recipe['ingredients'])
      items.append(recipe['name'])

  pca = Pipeline([
    ('vect', DictVectorizer(sparse=False)),
    ('pca', PCA(n_components=2))
  ])
  X_pca = pca.fit_transform(ingredients)
  labels = Pipeline([
    ('vect', DictVectorizer(sparse=False)),
    ('pca', PCA(n_components=2)),
    ('agglom', AgglomerativeWrapper(AgglomerativeClustering(n_clusters=6, linkage='ward')))
  ])

  labels.fit(ingredients)
  clusters = labels.named_steps['agglom'].labels_
  print(clusters)
  plt.figure()
  for row, item in enumerate(items):
    plt.scatter(X_pca[row, 0], X_pca[row, 1], s=100, c='rgbcyk'[clusters[row]])
    plt.annotate("{}".format(item),
                 xy=(X_pca[row, 0], X_pca[row, 1]),
                 textcoords='offset points',
                 xytext=(10,10),
                 size=10,
                 arrowprops=dict(arrowstyle="->",
                                 facecolor='white'))
  plt.show()


if __name__ == "__main__":
  main()
