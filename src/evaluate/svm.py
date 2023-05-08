from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class SVM():
    def __init__(self, C=[0.01,0.1,1,10,100],  cv=5):
        svc = SVC(random_state=42)
        self.classifier = GridSearchCV(svc, {"C": C}, cv=cv)

    def fit(self, train_embeddings, train_target):
        return self.classifier.fit(train_embeddings, train_target)

    def predict(self, test_embeddings):
        return self.classifier.predict(test_embeddings)
