
class Wrapper:
    def __init__(self, library, input_size, hidden_size, output_size):
        self.library = library
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = None
        self.key = None  # Initialize key for JAX

    def init_base_model(self):
        if self.library == "jax":
            import jax.numpy as jnp
            from jax import random
            
            self.key = random.PRNGKey(0)
            self.w1 = random.normal(self.key, (self.input_size, self.hidden_size))
            self.b1 = jnp.zeros(self.hidden_size)
            self.w2 = random.normal(self.key, (self.hidden_size, self.output_size))
            self.b2 = jnp.zeros(self.output_size)
        
        elif self.library == "tensorflow":
            from tensorflow import keras
            from tensorflow.keras import layers
            
            self.model = keras.Sequential([
                layers.Dense(self.hidden_size, activation='relu', input_shape=(self.input_size,)),
                layers.Dense(self.output_size)
            ])
            self.model.compile(
                optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

        elif self.library == "pytorch":
            import torch.nn as nn
            import torch.optim as optim
            
            self.model = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size)
            )
            self.loss_fn = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        return self.model
