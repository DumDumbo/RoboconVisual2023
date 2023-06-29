# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import os
import sys
from pathlib import Path
import time
import numpy
import torch
# import serial
from matplotlib.pyplot import box
from torch.backends import cudnn
import threading
from queue import Queue
"""
这段代码会获取当前文件的绝对路径，并使用Path库将其转换为Path对象。
"""
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (check_img_size, check_imshow, cv2,
                           non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadStreams, LoadImages
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box


"""
    * @ 功能：设置相机参数 
"""
# "视场角120°，在640x480中被分为640份,120/640=0.1875°，所以一份就是0.1875°"
camera_Angle = 0.1875
camera_Width = 640
camera_Height = 480
camera_Horizontal_View_Angle = 120

"""
    * @ 串口通信功能函数
"""




"""
    * @Module Function ：YoloV5 detect
    *
"""


def run(weights=ROOT / 'runs/train/exp4/weights/last.pt',
                   device='cpu',
                   imgsz=(640, 640),
                   half=False,
                   vid_stride=1,
                   conf_thres=0.65,
                   iou_thres=0.45,
                   line_thickness=3,
                   hide_labels=False,
                   hide_conf=False,
                   camera_id=id):
    """
    Description: Pre_Work
    """
    # Load model
    global det, center_x, x2, x1, target1, target2
    im0 = '0'
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    """
    Description:  Dataloader 、Warm up and Run inference
    """
    # Dataloader
    bs = 1
    dataset = LoadStreams(im0, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)

    # Warm up
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Run inference
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  # dt is for record time
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, max_det=1000)

        results = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            # batch_size >= 1
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '

            """
                *（119行至170行）
                *功能：          输出目标数据
                *数据输出类型：    Float
                *返回值：        返回所检测到目标的中心点距图像中心点的绝对值
3            """
            # Create a dictionary to store the data after processing
            target_order_dictionary = {0: [], 1: [], 2: [], 3: [],
                                       4: [], 5: [], 6: [], 7: [],
                                       8: [], 9: [], 10: []}

            # Create a list to store sorted data
            sorted_data = []

            # tensor object ===> list
            target = det.tolist()

            # start
            if target is not None:
                count = len(target)
                for y in range(0, count, 1):
                    sorted_data.append(
                        ((((target[y][0] + target[y][2]) / 2) - 320) / 320) * (camera_Horizontal_View_Angle / 2))

                sorted_data.sort()
                for g in range(0, count, 1):
                    target_order_dictionary[g] = sorted_data[g]

                """
                代码功能：将图像中心点坐标放入字典中

                for y in range(0, count, 1):
                    sorted_data.append(((target[y][0] + target[y][2]) / 2) - 320)
                '''-------------------------------------------------------'''
                sorted_data.sort()
                for g in range(0, count, 1):
                    target_order_dictionary[g] = sorted_data[g]
                '''--------------------------------------------------------'''
                """

                # ID = serial_communication_receive()
                # if ID:
                #     offset = target_order_dictionary.get(ID)
                #     print(offset)
                # else:
                #     pass

                # "接收stm32发来的选择信号存储在offset"
                # serial_communication_work(target_order_dictionary[])
                print(target_order_dictionary)
                # if len(target) == 0:
                #     pass
                #
                # elif len(target) == 1:
                #     target1 = ((target[0][0] + target[0][2]) / 2) - 320
                #     # 绝对值
                #     target1 = abs(target1)
                #
                #     print("Target1:", target1)

            else:
                pass

            # print("Original", det)
            # print("Target1:", target1)
            # det 储存了各个目标的（左上角xy坐标；右下角xy坐标；置信度；一个不知道是啥）
            # print效果： tensor（[[281,255,337,368,0.59,0.000]])  是一个二维张量tensor

        p = Path(p)  # to Path

        """
            * Description: Print location 、write result and stream result
        """

        # define class(Annotator) as annotator
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # write result
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))

        # Stream results
        im0 = annotator.result()
        windows.append(p)
        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.line(im0, (320, 0), (320, 480), (0, 255, 0))
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 millisecond


def main():
    run()


if __name__ == '__main__':
    main()
