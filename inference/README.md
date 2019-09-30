## C++ on-device (X86/ARM) inference app for YOLOv3 detection modelset

Here are some C++ implementation of the on-device inference for trained YOLOv3 inference model, including forward propagation of the model, YOLO postprocess and bounding box NMS. It support YOLOv3/Tiny YOLOv3 arch and all kinds of backbones & head. Now we have 2 different inference engine versions for that:

* Tensorflow-Lite
* [MNN](https://github.com/alibaba/MNN) from Alibaba


### MNN

1. Build libMNN and model convert tool

Refer to [MNN build guide](https://www.yuque.com/mnn/cn/build_linux). Since MNN support both X86 & ARM platform, we can do either native compile or ARM cross-compile
```
# apt install ocl-icd-opencl-dev
# git clone https://github.com/alibaba/MNN.git <Path_to_MNN>
# cd <Path_to_MNN>
# ./schema/generate.sh
# ./tools/script/get_model.sh  # optional
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] .. && make -j4

# cd ../tools/converter
# ./generate_schema.sh
# mkdir build && cd build && cmake .. && make -j4
```

2. Build demo inference application
```
# cd keras-YOLOv3-model-set/inference/MNN
# mkdir build && cd build
# cmake -DMNN_ROOT_PATH=<Path_to_MNN> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
```

3. Convert trained YOLOv3 model to MNN model

Refer to [Model dump](https://github.com/david8862/keras-YOLOv3-model-set#model-dump), [Tensorflow model convert](https://github.com/david8862/keras-YOLOv3-model-set#tensorflow-model-convert) and [MNN model convert](https://www.yuque.com/mnn/cn/model_convert), we need to:

* dump out inference model from training checkpoint:

    ```
    # python yolo.py --model_type=mobilenet_lite --model_path=logs/000/<checkpoint>.h5 --anchors_path=configs/tiny_yolo_anchors.txt --classes_path=configs/voc_classes.txt --model_image_size=320x320 --dump_model --output_model_file=model.h5
    ```

* convert keras .h5 model to tensorflow model (frozen pb):

    ```
    # python keras_to_tensorflow.py
        --input_model="path/to/keras/model.h5"
        --output_model="path/to/save/model.pb"
    ```

* convert TF pb model to MNN model:

    ```
    # cd <Path_to_MNN>/tools/converter/build
    # ./MNNConvert -f TF --modelFile model.pb --MNNModel model.pb.mnn --bizCode biz
    ```

4. Run application to do inference with model, or put all the assets to your ARM board and run if you use cross-compile
```
# cd keras-YOLOv3-model-set/inference/MNN/build
# ./yolov3Detection -h
Usage: yolov3Detection
--mnn_model, -m: model_name.mnn
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--anchors, -a: anchor values for the model
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--threads, -t: number of threads
--count, -c: loop model run for certain times
--warmup_runs, -w: number of warmup runs


# ./yolov3Detection -m model.pb.mnn -i ../../../example/dog.jpg -l ../../../configs/voc_classes.txt -a ../../../configs/tiny_yolo_anchors.txt -t 8 -c 10 -w 3
Can't Find type=4 backend, use 0 instead
image_input: w:320 , h:320, bpp: 3
num_classes: 20
origin image size: 768, 576
model invoke time: 43.015000 ms
output tensor name: conv2d_1/Conv2D
output tensor name: conv2d_3/Conv2D
Caffe format: NCHW
batch 0:
Caffe format: NCHW
batch 0:
yolo_postprocess time: 1.635000 ms
prediction_list size before NMS: 7
NMS time: 0.044000 ms
Detection result:
bicycle 0.779654 (145, 147) (541, 497)
car 0.929868 (471, 80) (676, 173)
dog 0.519254 (111, 213) (324, 520)
```
Here the [classes](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/configs/voc_classes.txt) & [anchors](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/configs/tiny_yolo_anchors.txt) file format are the same as used in training part


### Tensorflow-Lite

1. Build TF-Lite lib
We can do either native compile for X86 or cross-compile for ARM

```
# git clone https://github.com/tensorflow/tensorflow <Path_to_TF>
# cd <Path_to_TF>
# ./tensorflow/lite/tools/make/download_dependencies.sh
# make -f tensorflow/lite/tools/make/Makefile   #for X86 native compile
# ./tensorflow/lite/tools/make/build_rpi_lib.sh #for ARM cross compile, e.g Rasperberry Pi
```

2. Build demo inference application
```
# cd keras-YOLOv3-model-set/inference/tflite
# mkdir build && cd build
# cmake -DTF_ROOT_PATH=<Path_to_TF> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] [-DTARGET_PLAT=<target>] ..
# make
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" and "TARGET_PLAT" should be specified. Refer [CMakeLists.txt](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/inference/tflite/CMakeLists.txt) for details.

3. Convert trained YOLOv3 model to tflite model

Tensorflow-lite support both Float32 and UInt8 type model, so we can dump out the keras .h5 model to Float32 .tflite model or use [post_train_quant_convert.py](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/tools/post_train_quant_convert.py) script to convert to UInt8 model with TF 2.0 Post-training integer quantization tech, which could be smaller and faster on ARM:

* dump out inference model from training checkpoint:

    ```
    # python yolo.py --model_type=mobilenet_lite --model_path=logs/000/<checkpoint>.h5 --anchors_path=configs/tiny_yolo_anchors.txt --classes_path=configs/voc_classes.txt --model_image_size=320x320 --dump_model --output_model_file=model.h5
    ```

* convert keras .h5 model to Float32 tflite model:

    ```
    # tflite_convert --keras_model_file=model.h5 --output_file=model.tflite
    ```

* convert keras .h5 model to UInt8 tflite model with TF 2.0 Post-training integer quantization:

    ```
    # cd keras-YOLOv3-model-set/tools
    # python post_train_quant_convert.py --keras_model_file=model.h5 --annotation_file=<train/test annotation file to feed converter> --model_input_shape=320x320 --sample_num=30 --output_file=model_quant.tflite
    ```


4. Run TFLite validate script
```
# cd keras-YOLOv3-model-set/tools/
# python validate_yolo_tflite.py --model_path=model.tflite --anchors_path=../configs/tiny_yolo_anchors.txt --classes_path=../configs/voc_classes.txt --image_file=../example/dog.jpg --loop_count=5
```
#### You can also use [eval.py](https://github.com/david8862/keras-YOLOv3-model-set#evaluation) to do evaluation on the TFLite model



5. Run application to do inference with model, or put assets to ARM board and run if cross-compile
```
# cd keras-YOLOv3-model-set/inference/tflite/build
# ./yolov3Detection -h
Usage: yolov3Detection
--tflite_model, -m: model_name.tflite
--image, -i: image_name.jpg
--classes, -l: classes labels for the model
--anchors, -a: anchor values for the model
--input_mean, -b: input mean
--input_std, -s: input standard deviation
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--threads, -t: number of threads
--count, -c: loop interpreter->Invoke() for certain times
--warmup_runs, -w: number of warmup runs
--verbose, -v: [0|1] print more information

# ./yolov3Detection -m model.tflite -i ../../../example/dog.jpg -l ../../../configs/voc_classes.txt -a ../../../configs/tiny_yolo_anchors.txt -t 8 -c 10 -w 3 -v 1
Loaded model model.tflite
resolved reporter
num_classes: 20
origin image size: width:768, height:576, channel:3
input tensor info: type 1, batch 1, height 320, width 320, channels 3
invoked average time:107.479 ms
output tensor info: name conv2d_1/BiasAdd, type 1, batch 1, height 10, width 10, channels 75
batch 0
output tensor info: name conv2d_3/BiasAdd, type 1, batch 1, height 20, width 20, channels 75
batch 0
yolo_postprocess time: 3.618 ms
prediction_list size before NMS: 7
NMS time: 0.358 ms
Detection result:
bicycle 0.838566 (144, 141) (549, 506)
car 0.945672 (466, 79) (678, 173)
dog 0.597517 (109, 215) (326, 519)
```

### TODO
- [ ] Support MNN Quantized model
- [ ] further latency optimize on yolo postprocess C++ implementation
- [ ] refactor demo app to get common interface