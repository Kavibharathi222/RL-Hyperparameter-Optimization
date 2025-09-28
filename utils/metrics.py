from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    return {
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }
