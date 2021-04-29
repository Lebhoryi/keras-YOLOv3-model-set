import tensorflow as tf
import pathlib
import io
import PIL
import numpy as np


# Convert to a TensorFlow Lite model
def keras2tflite(h5_path):
    tflite_model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
    tflite_model = converter.convert()
    tflite_model_file = pathlib.Path('./weights/test.tflite')
    tflite_model_file.write_bytes(tflite_model)  # 84528
    print('Convert successfully!!!')


h5_path_list = ['weights/mnist.h5', 'weights/yolo-fastest.h5', 'weights/yolo-xl.h5', 'weights/yolo-l.h5']

# keras2tflite(h5_path_list[2])


def representative_dataset_gen():
    record_iterator = tf.python_io.tf_record_iterator(path='/home/lebhoryi/Data/visualwakewords/val.record-00000-of-00010')

    count = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        image_stream = io.BytesIO(example.features.feature['image/encoded'].bytes_list.value[0])
        image = PIL.Image.open(image_stream)
        image = image.resize((96, 96))
        image = image.convert('L')
        array = np.array(image)
        array = np.expand_dims(array, axis=2)
        array = np.expand_dims(array, axis=0)
        array = ((array / 127.5) - 1.0).astype(np.float32)
        yield([array])
        count += 1
        if count > 300:
            break

converter = tf.keras.models.load_model(h5_path_list[0])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
with open("test.tflite", "wb") as f:
    f.write(tflite_quant_model)