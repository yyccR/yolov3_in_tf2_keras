import sys

sys.path.append('../yolov3_in_tf2_keras')

import cv2
import os
import skimage.io as io
import numpy as np

from data.visual_ops import draw_bounding_box

from yolov3 import YoloV3


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # model_path = "h5模型路径, 默认在 ./logs/yolov3-tf-300.h5"
    model_path = "./logs/yolov3-tf-200.h5"
    # image_path = "提供你要测试的图片路径"
    image_path = "./data/coco_2017_val_images/289343.jpg"
    # image = cv2.imread(image_path)
    image = io.imread(image_path)[:,:,::-1]

    image_shape = (320, 320, 3)
    num_class = 91
    batch_size = 1

    # 这里anchor归一化到[0,1]区间
    anchors = np.array([[17, 20], [43, 52], [66, 127], [132, 69], [116, 243], [205, 149],
                        [233, 363], [410, 216], [496, 440]], np.float32) / 640.#image_shape[0]
    # 分别对应1/8, 1/16, 1/32预测输出层
    anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    # 自己训练集的类别
    # classes = ['_background_', 'yanghua', 'zhengchang']
    classes = ['none', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'none', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
               'bear', 'zebra', 'giraffe', 'none', 'backpack', 'umbrella', 'none', 'none', 'handbag',
               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'none', 'wine glass',
               'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'none',
               'dining table', 'none', 'none', 'toilet', 'none', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'none', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    yolo = YoloV3(
        weights_path=model_path,
        image_shape=image_shape,
        batch_size=batch_size,
        num_class=num_class,
        is_training=False,
        anchors=anchors,
        anchor_masks=anchor_masks,
    )
    # yolo.yolo_model.summary(line_length=100)
    #  待预测照片的路径

    predicts = yolo.predict(image)
    # print(predicts)
    # 预测出的结果有四类分别为边框 分数 类别 总数
    boxes, scores, class_ids, num = predicts
    print(boxes)
    print(scores)
    # print(type(boxes))
    # boxes = np.array(boxes)
    # scores = np.array(scores)
    # class_ids = np.array(class_ids)
    # 新建一个路径
    if not os.path.isdir("./data/tmp"):
        os.mkdir("./data/tmp")
    pred_image = image.copy()
    if num[0][0]:
        for j in range(num[0][0]):
            if scores[0][j] > 0.5:
                label = int(class_ids[0][j])
                class_name = classes[label]
                xmin, ymin, xmax, ymax =boxes[0][j]
                pred_image = draw_bounding_box(pred_image, class_name, scores[0][j], int(xmin), int(ymin),
                                               int(xmax), int(ymax))
    #上面的操作是将边框 分数 类别 画到图片上
    cv2.imwrite("./data/tmp/predicts.jpg", pred_image)
    # cv2.imshow("prediction", pred_image)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()