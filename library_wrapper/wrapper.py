
class Wrapper:
    def __init__(self, library, input_size, hidden_size, output_size, model_type='feedforward'):
        self.library = library
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.model = None
        self.criterion = None


    def init_default_model(self):
        if self.library == "jax":
            self._init_jax_model()
        elif self.library == "tensorflow":
            self._init_tensorflow_model()
        elif self.library == "pytorch":
            self._init_pytorch_model()
        return self.model

    def _init_jax_model(self):
        import jax.numpy as jnp
        from jax import random

        self.key = random.PRNGKey(0)
        if self.model_type == 'feedforward':
            self.w1 = random.normal(self.key, (self.input_size, self.hidden_size))
            self.b1 = jnp.zeros(self.hidden_size)
            self.w2 = random.normal(self.key, (self.hidden_size, self.output_size))
            self.b2 = jnp.zeros(self.output_size)

    def _init_tensorflow_model(self):
        from tensorflow import keras
        from tensorflow.keras import layers

        if self.model_type == 'feedforward':
            self.model = keras.Sequential([
                layers.Dense(self.hidden_size, activation='relu', input_shape=(self.input_size,)),
                layers.Dense(self.output_size)
            ])
        elif self.model_type == 'recurrent':
            self.model = keras.Sequential([
                layers.LSTM(self.hidden_size, input_shape=(None, self.input_size)),
                layers.Dense(self.output_size)
            ])
        elif self.model_type == 'convolutional':
            self.model = keras.Sequential([
                layers.Conv2D(self.hidden_size, kernel_size=(3, 3), activation='relu',
                              input_shape=(self.input_size, self.input_size, 1)),
                layers.Flatten(),
                layers.Dense(self.output_size)
            ])
        self.model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def _init_pytorch_model(self):
        import torch.nn as nn
        import torch.optim as optim

        if self.model_type == 'feedforward':
            self.model = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.output_size)
            )
        elif self.model_type == 'recurrent':
            self.model = nn.LSTM(self.input_size, self.hidden_size)
            self.fc = nn.Linear(self.hidden_size, self.output_size)
            
        elif self.model_type == 'convolutional':
            self.model = nn.Sequential(
                nn.Conv2d(1, self.hidden_size, kernel_size=3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self.hidden_size * ((self.input_size - 2) ** 2), self.output_size)
            )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


    def get_criterion(self):
        return self.loss_fn
    
    
    def get_optimizer(self):
        return self.optimizer
    

    def default_model(self):
        return self.init_default_model()


