@echo off
CHCP 65001
color a
echo 开始训练比赛模型 Faster R-CNN
python train_frcnn.py --path="./data/annotation.txt" --network="vgg"  --input_weight_path="./pre_train/vgg16_weights_tf_kernels_notop.h5"
rem python train_frcnn.py --path="./train/annotation_1.csv" --network="vgg"


Pause

rem # 命令行参数
rem parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
rem parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
rem                   default="QD_CNN")
rem parser.add_option("-n", "--num_rois", dest="num_rois", help="Number of RoIs to process at once.", default=32)
rem parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
rem parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
rem parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
rem parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
rem                   action="store_true", default=False)
rem parser.add_option("--num_epochs", dest="num_epochs", help="Number of epochs.", default=2000)
rem parser.add_option("--config_filename", dest="config_filename",
rem                   help="Location to store all the metadata related to the training (to be used when testing).",
rem                   default="config.pickle")
rem parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
rem parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
