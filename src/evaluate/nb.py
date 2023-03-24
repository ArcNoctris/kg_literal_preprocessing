from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV





class NB():
    def __init__(self, random_state=42, alpha=[1, 2, 4], cv=10):
        nb = CategoricalNB(random_state=random_state)
        self.classifier = GridSearchCV(nb, {"alpha": alpha}, cv=cv)

    def fit(self, train_embeddings, train_target):
        return self.classifier.fit(train_embeddings, train_target)

    def predict(self, test_embeddings):
        return self.classifier.predict(test_embeddings)
