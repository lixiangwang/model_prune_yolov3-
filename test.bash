CUDA_VISIBLE_DEVICES=3 python test.py --cfg cfg/prune_10_shortcut_prune_0.8_keep_0.01_dense_yolov3_4.cfg --data data/visdrone.data --weights weights/prune_10_shortcut_prune_0.8_keep_0.01_best.weights --batch-size 64 --img-size 800 


