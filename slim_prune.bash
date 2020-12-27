# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --cfg cfg/dense_yolov3_4.cfg --data data/visdrone.data --weights weights/yolov3.weights --epoch 120 --batch-size 4
CUDA_VISIBLE_DEVICES=0 python slim_prune.py --cfg cfg/dense_yolov3_4.cfg --data data/visdrone.data --weights weights/best.pt --global_percent 0.8 --layer_keep 0.01 --img_size 800

