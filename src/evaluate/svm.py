from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class SVM():
    def __init__(self, random_state=42, C=[0.01,0.1,1,10,100,1000],  cv=10):
        svc = SVC(random_state=random_state)
        self.classifier = GridSearchCV(svc, {"C": C}, cv=cv)

    def fit(self, train_embeddings, train_target):
        return self.classifier.fit(train_embeddings, train_target)

    def predict(self, test_embeddings):
        return self.classifier.predict(test_embeddings)
