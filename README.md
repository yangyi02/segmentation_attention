Suppose the codes are located at deeplab/code

1. Create a folder for experiments at deeplab/exper

2.1 Create a folder for your specific experiment, let's take PASCAL VOC 2012 for example, at deeplab/exper/voc12

2.2 Create folders for voc12:
deeplab/exper/voc12/config      : where network config files are saved
deeplab/exper/voc12/features  : where the computed features will be saved (when train on train)
deeplab/exper/voc12/features2 : where the computed features will be saved (when train on trainval)
deeplab/exper/voc12/list : where you save the train, val, and test file lists
deeplab/exper/voc12/log : where the training/test logs will be saved
deeplab/exper/voc12/model : where the trained models will be saved
deeplab/exper/voc12/res : where the evaluation results will be saved

3.1 Test your own network: create a folder under config. For example, config/deeplab_largeFOV, where deeplab_largeFOV is the network you want to experiment with. Add your train.prototxt and test.prototxt in that folder (you can check some provided examples for reference) .

3.2 Set up your init.caffemodel at model/deeplab_largeFOV. You may want to soft link init.caffemodel to the modified VGG-16 net.

4.1 Then, modify the provided script for experiments: run_pascal.sh, where you should change the paths according to your setting. For example, you should specify where the caffe is by changing CAFFE_DIR.

4.2 You may need to modify sub.sed, if you want to replace some variables with your desired values in train.prototxt or test.prototxt.

5. The computed features are saved at folders features or features2, and you can run provided MATLAB scripts to evaluate the results (e.g., check the script at code/matlab/my_script/EvalSegResults).