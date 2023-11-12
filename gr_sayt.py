"""
Powered by @x4cc3r
"""
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import gradio as gr

model = load_model("keras_model.h5")

labels = open("labels.txt", "r").readlines()


def classify_image(inp):
    inp = tf.image.resize(inp, (224, 224))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = model.predict(tf.expand_dims(inp, axis=0))[0]
    return {labels[i]: float(prediction[i]) for i in range(6)}

gr.Interface(fn=classify_image, inputs="image", outputs="label").launch()