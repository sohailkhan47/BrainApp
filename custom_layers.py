# custom_layers.py
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K

class FixedDropout(Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        return super(FixedDropout, self).call(inputs, training)

# Register the custom layer globally
get_custom_objects().update({'FixedDropout': FixedDropout})
