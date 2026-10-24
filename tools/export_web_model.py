"""Export the trained Keras model for the browser version in docs/.

    python tools/export_web_model.py app/export/wastevision_v2.keras docs/model

Writes backbone.onnx (EfficientNetV2-B0 feature extractor, input 1x224x224x3 in 0..255,
output 1x7x7x1280) and head.bin (float32: W1 1280x128, b1 128, W2 128x11, b2 11).
Requires tensorflow, keras and tf2onnx.
"""
import os, sys
import numpy as np
import tensorflow as tf
import keras
import tf2onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
model = keras.saving.load_model(src, compile=False)
backbone, head = model.get_layer("backbone"), model.get_layer("head")
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image"),)

@tf.function(input_signature=spec)
def features(x):
    return {"features": backbone(x, training=False)}

frozen = convert_variables_to_constants_v2(features.get_concrete_function())   # bake normalisation constants in
tf2onnx.convert.from_graph_def(frozen.graph.as_graph_def(), input_names=[i.name for i in frozen.inputs],
                               output_names=[o.name for o in frozen.outputs], opset=17,
                               output_path=os.path.join(out, "backbone.onnx"))
W1, b1 = head.get_layer("head_dense").get_weights()
W2, b2 = head.get_layer("logits").get_weights()
np.concatenate([W1.ravel(), b1, W2.ravel(), b2]).astype("<f4").tofile(os.path.join(out, "head.bin"))
print("wrote", os.listdir(out))
