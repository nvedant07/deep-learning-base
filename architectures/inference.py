import timm

def inference_with_features(model, X):
    model.forward(X)