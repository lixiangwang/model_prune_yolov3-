CUDA_VISIBLE_DEVICES=3 python train.py --cfg cfg/dense_yolov3_4.cfg --data data/visdrone.data --weights weights/best.weights --epoch 300 --batch-size 16 --sr_mode 1 --s 0.003


