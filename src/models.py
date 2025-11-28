import numpy as np


# 1. Logistic Regression core
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


class LogisticRegressionNumpy:
    """
    Logistic Regression cài bằng NumPy, dùng gradient descent.
    """

    def __init__(self, lr: float = 0.01, n_iter: int = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.loss_history = []

        for i in range(self.n_iter):
            # logits = X @ w + b
            logits = np.einsum('ij,j->i', X, self.w) + self.b
            y_pred = sigmoid(logits)

            error = y_pred - y
            # dw = X^T (y_pred - y) / N
            dw = np.einsum('ij,i->j', X, error) / n_samples
            db = float(np.mean(error))

            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            if verbose and ((i + 1) % 100 == 0 or i == 0):
                print(f"Iter {i+1}/{self.n_iter} - loss: {loss:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.einsum('ij,j->i', X, self.w) + self.b
        return sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


# 2. Metrics & confusion matrix
def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp, tn, fp, fn = confusion_matrix_np(y_true, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp + 1e-15)
    rec = tp / (tp + fn + 1e-15)
    f1 = 2 * prec * rec / (prec + rec + 1e-15)
    return acc, prec, rec, f1, (tp, tn, fp, fn)



