# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --cfg cfg/dense_yolov3_4.cfg --data data/visdrone.data --weights weights/yolov3.weights --epoch 120 --batch-size 4
CUDA_VISIBLE_DEVICES=0 python layer_prune.py --cfg cfg/prune_0.8_keep_0.01_dense_yolov3_4.cfg --data data/visdrone.data --weights weights/prune_0.8_keep_0.01_best.weights --shortcuts 10 --img_size 800

