# CUDA_VISIBLE_DEVICES=0 python train.py --cfg cfg/dense_yolov3_4.cfg --data data/visdrone.data --weights weights/yolov3.weights --epoch 120 --batch-size 16
CUDA_VISIBLE_DEVICES=0 python train.py --cfg cfg/prune_10_shortcut_prune_0.8_keep_0.01_dense_yolov3_4.cfg --data data/visdrone.data --weights weights/prune_10_shortcut_prune_0.8_keep_0.01_best.weights --epoch 150 --batch-size 16
