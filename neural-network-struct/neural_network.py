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
