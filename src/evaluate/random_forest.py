from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForest():
    def __init__(self, random_state=42, n_estimators=[10, 20, 40], max_depth=[3, 5, 10, None], cv=10):
        gscv = RandomForestClassifier(random_state=random_state)
        self.classifier = GridSearchCV(gscv, {
                                       "n_estimators": n_estimators, 
                                       "max_depth": max_depth, 
                                       "random_state": random_state}, 
                                       cv=cv)

    def fit(self, train_embeddings, train_target):
        return self.classifier.fit(train_embeddings, train_target)

    def predict(self, test_embeddings):
        return self.classifier.predict(test_embeddings)
