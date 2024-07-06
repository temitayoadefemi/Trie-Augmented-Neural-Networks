import torch
import jax.numpy as jnp
import tensorflow as tf

class Tensor:
    def __init__(self, data, backend='pytorch'):
        self.backend = backend
        if backend == 'pytorch':
            self.data = torch.tensor(data)
        elif backend == 'jax':
            self.data = jnp.array(data)
        elif backend == 'tensorflow':
            self.data = tf.convert_to_tensor(data)
        else:
            raise ValueError("Unsupported backend")

    def add(self, other):
        if self.backend == 'pytorch':
            return Tensor(self.data + other.data, backend='pytorch')
        elif self.backend == 'jax':
            return Tensor(self.data + other.data, backend='jax')
        elif self.backend == 'tensorflow':
            return Tensor(self.data + other.data, backend='tensorflow')

    def multiply(self, other):
        if self.backend == 'pytorch':
            return Tensor(self.data * other.data, backend='pytorch')
        elif self.backend == 'jax':
            return Tensor(self.data * other.data, backend='jax')
        elif self.backend == 'tensorflow':
            return Tensor(self.data * other.data, backend='tensorflow')

    def __getitem__(self, index):
        if self.backend == 'pytorch':
            return self.data[index]
        elif self.backend == 'jax':
            return self.data[index]
        elif self.backend == 'tensorflow':
            return self.data.numpy()[index]  # TensorFlow tensors are not directly indexable without converting to numpy

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        return len(self.data)
