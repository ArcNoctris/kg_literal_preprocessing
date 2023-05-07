from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class KNN():
    def __init__(self, n_neighbors=[3, 5, 8,15], leaf_size=[15, 30, 45], cv=5):
        knn = KNeighborsClassifier()
        self.classifier = GridSearchCV(knn, {"n_neighbors": n_neighbors, 'leaf_size':leaf_size}, cv=cv)

    def fit(self, train_embeddings, train_target):
        return self.classifier.fit(train_embeddings, train_target)

    def predict(self, test_embeddings):
        return self.classifier.predict(test_embeddings)