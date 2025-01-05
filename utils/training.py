'''
import tensorflow as tf
from keras.models import Model
from consts import *
import numpy as np


def training_loop(model: Model, loss, opt, epochs, dataset) -> Model:
    for epoch in range(epochs):
        for step, (inp, tar) in enumerate(dataset):
            with tf.GradientTape() as tape:
                logits = model(inp, training=true)
                c_loss = loss(tar, logits)

            grads = tape.gradient(c_loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
    return model
'''
