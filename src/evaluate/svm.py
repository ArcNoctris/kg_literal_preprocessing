from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV





class SVM():
    def __init__(self, random_state=42, n_estimators=[10, 20, 40], max_depth=[3, 5, None], cv=10):
        svc = SVC(random_state=random_state)
        self.classifier = GridSearchCV(svc, {"C": [10**i for i in range(-3, 4)]}, cv=cv)

    def fit(self, train_embeddings, train_target):
        return self.classifier.fit(train_embeddings, train_target)

    def predict(self, test_embeddings):
        return self.classifier.predict(test_embeddings)
