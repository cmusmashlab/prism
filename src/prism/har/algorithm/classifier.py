import pickle
from sklearn.ensemble import RandomForestClassifier


class Classifier():

    def __init__(self):
        self.model = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
        print('Classifier initialized.')

    def train(self, X, y):
        self.model.fit(X, y)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print('Classifier loaded from ', path)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)