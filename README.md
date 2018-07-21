# 3D Human Pose Estimation with Relational Networks
Source code for the paper '[3D Human Pose Estimation with Relational Networks](https://arxiv.org/abs/1805.08961)', BMVC 2018

Video results can be found [here](https://www.youtube.com/watch?v=JIeDtnNLOdc).

## Installation

First, install `Caffe` and `MatCaffe` from the official [Caffe repository](https://github.com/BVLC/caffe) following [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html). This code has been tested on Ubuntu and Windows OS.

To use the relational dropout, you need to register RNDropoutLayer to your Caffe build. We will denote `$CAFFE_ROOT` as the root directory of the Caffe code. Place the `rn_dropout_layer.hpp` to `$CAFFE_ROOT/include/caffe/layers/` and `rn_dropout_layer.cpp` and `rn_dropout_layer.cu` to `$CAFFE_ROOT/src/caffe/layers/`. Next, the parameters for RNDropout layer should be specified in `caffe.proto`. Add `optional RNDropoutParameter rn_dropout_param = (next_available_id);` (line 414 of `caffe.proto.example`) to your proto file with appropriate `next_available_id` value. Then, copy line 418-435 of `caffe.proto.example` to your proto file, and build the project.

## Download Datasets

The experiments are conducted on the [Human 3.6m dataset](http://vision.imar.ro/human3.6m/). We provided the 2D pose estimation results of training and testing images using [Stacked Hourglass Networks](https://github.com/umich-vl/pose-hg-demo), which are used for our experiments. Run `./download_dataset.sh` in the project root folder. This will generate two directories: `data_train` which is in HDF-5 format, and `data_test` which contains `.mat` files used in MATLAB.

## Download Pre-trained Models

Pre-trained models are provided to reproduce the results in the paper. Run `./download_models.sh` to download pre-trained models. Two pre-trained models are provided: the caffemodel in `/models/RN/` directory will generate the results of `RN` in Table 1 in the paper, the other in `/models/RN_drop/` directory will generate the results of `RN-hier-drop` in Table 3 in the paper.

## Training

To train the network without relational dropout, run
```
$CAFFE_ROOT/build/tools/caffe train -gpu=0 -solver=models/RN/solver_rn.prototxt
```
in the project root folder.

To train the model with relational dropout, run

```
$CAFFE_ROOT/build/tools/caffe train -gpu=0 -solver=models/RN_drop/solver_rn.prototxt
```


## Evaluation

To evaluate 3D pose estimation performance on Human3.6M test dataset, go to `matlab_src` directory in MATLAB and run
```
test_all(model_dir,deploy_model_name,iterations)
```
For instance, to evaluate RN model without relational dropout run `test_all('../models/RN/','net_test',100000)`. For the RN_drop model, we provided several deploy models to simulate various kinds of occlusion. To measure the performance when the joints of right leg are missing, run `test_all('../models/RN_drop/','net_test_right_leg_missing',100000)`
