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


class Wrapper():
    def __init__(self, library):
        self.library = library


    def init_base_model(self, library, model):
        self.model = model
        if library == "jax":
            self.model = random.PRNGKey(0)
            self.w1 = random.normal(self.key, (self.input_size, self.hidden_size))
            self.b1 = jnp.zeros(self.hidden_size)
            self.w2 = random.normal(self.key, (self.hidden_size, self.output_size))
            self.b2 = jnp.zeros(self.output_size)

        if library == "tensorflow":
            self.model = keras.Sequential([
                layers.Dense(self.hidden_size, activation='relu', input_shape=(self.input_size,)),
                layers.Dense(self.output_size)
            ])
            self.model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        if library == "pytorch":
            self.model = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size)
            )
            self.loss_fn = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        return model