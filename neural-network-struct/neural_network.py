import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TrieNetwork:
    def __init__(self, framework='pytorch', input_size=784, hidden_size=128, output_size=10):
        self.framework = framework
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        if self.framework == 'jax':
            self.init_jax_model()
        elif self.framework == 'tensorflow':
            self.init_tf_model()
        elif self.framework == 'pytorch':
            self.init_torch_model()
        else:
            raise ValueError("Unsupported framework")

    # JAX initialization

    # Training
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        if self.framework == 'jax':
            self.train_jax(X_train, y_train, epochs, batch_size)
        elif self.framework == 'tensorflow':
            self.train_tf(X_train, y_train, epochs, batch_size)
        elif self.framework == 'pytorch':
            self.train_torch(X_train, y_train, epochs, batch_size)

    def train_jax(self, X_train, y_train, epochs, batch_size):
        @jit
        def predict(params, x):
            w1, b1, w2, b2 = params
            x = jnp.dot(x, w1) + b1
            x = jax.nn.relu(x)
            x = jnp.dot(x, w2) + b2
            return x

        @jit
        def loss(params, x, y):
            preds = predict(params, x)
            return jnp.mean(jax.nn.sparse_softmax_cross_entropy_with_logits(y, preds))

        @jit
        def update(params, x, y, lr=0.001):
            grads = grad(loss)(params, x, y)
            return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]

        params = [(self.w1, self.b1), (self.w2, self.b2)]
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = jnp.array(X_train[i:i+batch_size])
                y_batch = jnp.array(y_train[i:i+batch_size])
                params = update(params, X_batch, y_batch)

    def train_tf(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def train_torch(self, X_train, y_train, epochs, batch_size):
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

    # Evaluation
    def evaluate(self, X_test, y_test):
        if self.framework == 'jax':
            return self.evaluate_jax(X_test, y_test)
        elif self.framework == 'tensorflow':
            return self.evaluate_tf(X_test, y_test)
        elif self.framework == 'pytorch':
            return self.evaluate_torch(X_test, y_test)

    def evaluate_jax(self, X_test, y_test):
        @jit
        def predict(params, x):
            w1, b1, w2, b2 = params
            x = jnp.dot(x, w1) + b1
            x = jax.nn.relu(x)
            x = jnp.dot(x, w2) + b2
            return x

        params = [(self.w1, self.b1), (self.w2, self.b2)]
        preds = predict(params, jnp.array(X_test))
        accuracy = jnp.mean(jnp.argmax(preds, axis=1) == jnp.array(y_test))
        return accuracy

    def evaluate_tf(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return accuracy

    def evaluate_torch(self, X_test, y_test):
        with torch.no_grad():
            outputs = self.model(torch.tensor(X_test, dtype=torch.float32))
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == torch.tensor(y_test, dtype=torch.long)).float().mean()
        return accuracy

    # Prediction
    def predict(self, X):
        if self.framework == 'jax':
            return self.predict_jax(X)
        elif self.framework == 'tensorflow':
            return self.predict_tf(X)
        elif self.framework == 'pytorch':
            return self.predict_torch(X)

    def predict_jax(self, X):
        @jit
        def predict(params, x):
            w1, b1, w2, b2 = params
            x = jnp.dot(x, w1) + b1
            x = jax.nn.relu(x)
            x = jnp.dot(x, w2) + b2
            return x

        params = [(self.w1, self.b1), (self.w2, self.b2)]
        preds = predict(params, jnp.array(X))
        return jnp.argmax(preds, axis=1)

    def predict_tf(self, X):
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)

    def predict_torch(self, X):
        with torch.no_grad():
            outputs = self.model(torch.tensor(X, dtype=torch.float32))
            _, preds = torch.max(outputs, 1)
        return preds.numpy()

# Example usage:
# X_train, y_train, X_test, y_test = ... # Load your data here
# model = TrieNetwork(framework='tensorflow', input_size=784, hidden_size=128, output_size=10)
# model.train(X_train, y_train, epochs=10, batch_size=32)
# accuracy = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {accuracy}")


        
