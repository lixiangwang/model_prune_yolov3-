
import os

import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *



parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./cfg/dense_yolov3_4.cfg', help='*.cfg path')
#parser.add_argument('--cfg', type=str, default='./cfg/prune_10_shortcut_prune_0.8_keep_0.01_dense_yolov3_4.cfg', help='*.cfg path')
parser.add_argument('--img_path', type=str, default='./data/car/', help='*.img path')
parser.add_argument('--names', type=str, default='./data/visdrone.names', help='*.names path')
parser.add_argument('--weights', type=str, default='./weights/sparse_421map_best.pt', help='weights path')
#parser.add_argument('--weights', type=str, default='./weights/last.pt', help='weights path')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
opt = parser.parse_known_args()[0]

device = torch_utils.select_device(device=opt.device)
class model_pred(object):
    def __init__(self, opt=opt, device=device):
        super(model_pred, self).__init__()
        self.opt = opt
        self.device = device

    def load_model(self, weight_path, cfg_path, imgsz):
        model = Darknet(cfg_path, imgsz)
        model.load_state_dict(torch.load(weight_path, map_location=self.device)['model'])
        model.to(self.device).eval()
        return model

    def get_pred(self, img, model):
        with torch.no_grad():
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                       multi_label=False, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

        return pred

    def img_process(self, img_dir_update):
        # 对图像的处理
        img0 = cv2.imread(img_dir_update)
        img = letterbox(img0, new_shape=640)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)  # to GPU
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img, img0


class Inference(model_pred):

    def __init__(self):
        super(Inference, self).__init__()
        self.img_dir = self.opt.img_path
        self.weight_dir = self.opt.weights
        self.cfg_dir = self.opt.cfg

        self.model = self.load_model(self.weight_dir, self.cfg_dir, self.opt.img_size)

    def infer_toresult(self):
        path = self.img_dir
        file_path = os.listdir(path)
        file_path.sort(key = lambda x:int(x[:-4]))
        for img_path in file_path:
            name=img_path[:-4]
            img, img0 = self.img_process(os.path.join(path, img_path))
            t1 = torch_utils.time_synchronized()
            pred = self.get_pred(img, self.model)
            t2 = torch_utils.time_synchronized()
#            print('inference time:',t2-t1)
            names = load_classes(self.opt.names)
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]



            for i, det in enumerate(pred):

                s, im0 = '', img0
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from imgsz to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                    cv2.imshow('windows',im0)
                    cv2.waitKey(1)



if __name__ == '__main__':

    while True:
        infer = Inference()
        infer.infer_toresult()


