from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #type: ignore
import torch #type:ignore

def evaluate(model, X_test, y_test):

    model.eval()

    with torch.no_grad():

        logits = model(X_test)
        probs = torch.sigmoid(logits)

        preds = (probs > 0.5).int()

    y_true = y_test.numpy()
    y_pred = preds.numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }