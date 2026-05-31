class GroundTruthModel:
    def __init__(self, f):
        self.f = f

    def predict(self, X):
        return self.f(X)
