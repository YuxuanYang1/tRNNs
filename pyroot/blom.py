#blom.py
import numpy as np
import os
import csv

class XORNet:
    def __init__(self, layers, lr=0.1, seed=42):
        np.random.seed(seed)
        self.layers = layers
        self.lr = lr
        self.W = []
        self.b = []
        for i in range(len(layers) - 1):
            std = np.sqrt(2.0 / layers[i])
            self.W.append(np.random.randn(layers[i], layers[i+1]) * std)
            self.b.append(np.zeros((1, layers[i+1])))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        a = X
        n = len(self.W)
        for i in range(n):
            z = a @ self.W[i] + self.b[i]
            self.z_values.append(z)
            if i == n - 1:
                a = self.sigmoid(z)
            else:
                a = self.relu(z)
            self.activations.append(a)
        return a

    def backward(self, X, y):
        m = X.shape[0]
        n = len(self.W)
        delta = self.activations[-1] - y
        for i in range(n - 1, -1, -1):
            dW = self.activations[i].T @ delta / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            if i > 0:
                delta = (delta @ self.W[i].T) * (self.z_values[i-1] > 0)
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    @staticmethod
    def compute_loss(y_pred, y_true):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def save(self, path):
        arrays = {}
        for i, w in enumerate(self.W):
            arrays[f"W{i}"] = w
        for i, bb in enumerate(self.b):
            arrays[f"b{i}"] = bb
        arrays["layers"] = np.array(self.layers)
        np.savez(path,** arrays)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        n = (len(data.files) - 1) // 2
        self.W = [data[f"W{i}"] for i in range(n)]
        self.b = [data[f"b{i}"] for i in range(n)]
        self.layers = list(data["layers"])

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)
