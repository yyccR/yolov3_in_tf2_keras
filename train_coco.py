import sys

sys.path.append('../yolov3_in_tf2_keras')

import os
import numpy as np
import tensorflow as tf
from data_ops import transform_targets
from loss import loss
from data.generate_coco_data import CoCoDataGenrator
from data.visual_ops import draw_bounding_box
from yolov3 import YoloV3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    epochs = 301
    num_class = 91
    image_shape = [640, 640, 3]
    is_training = True
    batch_size = 4
    # -1表示全部数据参与训练
    train_img_nums = -1

    anchors = np.array([[17, 20], [43, 52], [66, 127], [132, 69], [116, 243], [205, 149],
                        [233, 363], [410, 216], [496, 440]], np.float32) / 640. #image_shape[0]
    # anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
    #                          (59, 119), (116, 90), (156, 198), (373, 326)],
    #                         np.float32) / self.image_shape[0]
    anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    # coco数据
    coco_annotation_file = "./data/instances_val2017.json"
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
    train_data = CoCoDataGenrator(
        coco_annotation_file=coco_annotation_file,
        img_shape=image_shape,
        batch_size=batch_size,
        max_instances=100,
        train_img_nums=train_img_nums,
        include_mask=False,
        include_crowd=False,
        include_keypoint=False,
        need_down_image=True
    )

    # tensorboard日志
    log_dir = "./logs"
    summary_writer = tf.summary.create_file_writer(log_dir)

    # 初始化模型
    yolo3 = YoloV3(num_class=num_class, batch_size=batch_size, is_training=is_training,
                   anchors=anchors, anchor_masks=anchor_masks)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(epochs):
        if epoch % 20 == 0 and epoch != 0:
            yolo3.yolo_model.save_weights(log_dir + '/yolov3-tf-{}.h5'.format(epoch))
        for batch in range(train_data.total_batch_size):
            with tf.GradientTape() as tape:
                data = train_data.next_batch()
                gt_imgs = data['imgs'] / 255.
                gt_boxes = data['bboxes'] / image_shape[0]
                gt_classes = data['labels']
                valid_nums = data['valid_nums']

                print("-------epoch {}, step {}, total step {}--------".format(epoch, batch,
                                                                               epoch * train_data.total_batch_size + batch))
                print("current data index: ",
                      train_data.img_ids[(train_data.current_batch_index - 1) * train_data.batch_size:
                                        train_data.current_batch_index * train_data.batch_size])
                for i, nums in enumerate(valid_nums):
                    print("gt boxes: ", gt_boxes[i, :nums, :] * image_shape[0])
                    print("gt classes: ", gt_classes[i, :nums])

                yolo_targets = transform_targets(
                    gt_boxes=gt_boxes,
                    gt_lables=gt_classes,
                    anchors=anchors,
                    anchor_masks=anchor_masks,
                    im_size=image_shape[0]
                )
                yolo_preds = yolo3.yolo_model(gt_imgs, training=True)

                # 3层输出分别计算损失
                total_xy_loss = total_wh_loss = total_obj_loss = total_class_loss = 0.
                for i in range(3):
                    true_box, true_obj, true_class = np.split(yolo_targets[i], (4, 5), axis=-1)
                    pred_box, pred_obj, pred_class, pred_box_xywh = yolo_preds[i]

                    xy_loss, wh_loss, obj_loss, class_loss = loss(
                        pred_box=pred_box,
                        pred_box_xywh=pred_box_xywh,
                        true_box=true_box,
                        pred_obj=pred_obj,
                        true_obj=true_obj,
                        pred_class=pred_class,
                        true_class=true_class,
                        anchors=anchors[anchor_masks[i]],
                        ignore_thresh=0.5
                    )
                    # print(i, tf.reduce_mean(xy_loss),  tf.reduce_mean(obj_loss))

                    total_xy_loss += tf.reduce_mean(xy_loss)
                    total_wh_loss += tf.reduce_mean(wh_loss)
                    total_obj_loss += tf.reduce_mean(obj_loss)
                    total_class_loss += tf.reduce_mean(class_loss)

                total_loss = total_xy_loss + total_wh_loss + total_obj_loss + total_class_loss
                grad = tape.gradient(total_loss, yolo3.yolo_model.trainable_variables)
                optimizer.apply_gradients(zip(grad, yolo3.yolo_model.trainable_variables))

                # Scalar
                with summary_writer.as_default():
                    tf.summary.scalar('loss/xy_loss', total_xy_loss,
                                      step=epoch * train_data.total_batch_size + batch)
                    # step=step)
                    tf.summary.scalar('loss/wh_loss', total_wh_loss,
                                      step=epoch * train_data.total_batch_size + batch)
                    # step=step)
                    tf.summary.scalar('loss/obj_loss', total_obj_loss,
                                      step=epoch * train_data.total_batch_size + batch)
                    # step=step)
                    tf.summary.scalar('loss/class_loss', total_class_loss,
                                      step=epoch * train_data.total_batch_size + batch)
                    # step=step)
                    tf.summary.scalar('loss/total_loss', total_loss,
                                      step=epoch * train_data.total_batch_size + batch)
                    # step=step)

                # image, 只拿每个batch的第一张
                # gt
                gt_img = gt_imgs[0].copy() * 255
                gt_boxes = gt_boxes[0] * image_shape[0]
                gt_classes = gt_classes[0]
                non_zero_ids = np.where(np.sum(gt_boxes, axis=-1))[0]
                for i in non_zero_ids:
                    label = gt_classes[i]
                    class_name = classes[label]
                    xmin, ymin, xmax, ymax = gt_boxes[i]
                    gt_img = draw_bounding_box(gt_img, class_name, label, int(xmin), int(ymin), int(xmax),
                                               int(ymax))

                # pred
                pred_img = gt_imgs[0].copy() * 255
                boxes, scores, class_ids, valid_detection_nums = yolo3.yolo_nms(yolo_preds, num_class)
                # print(scores)
                # print(gt_classes)
                for i in range(valid_detection_nums[0]):
                    if scores[0][i] > 0.5:
                        label = class_ids[0][i].numpy()
                        class_name = classes[label]
                        xmin, ymin, xmax, ymax = boxes[0][i] * image_shape[0]
                        pred_img = draw_bounding_box(pred_img, class_name, scores[0][i], int(xmin), int(ymin),
                                                     int(xmax), int(ymax))

                concat_imgs = tf.concat([gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1)
                summ_imgs = tf.expand_dims(concat_imgs, 0)
                summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                with summary_writer.as_default():
                    tf.summary.image("imgs/gt,pred,epoch{}".format(epoch), summ_imgs,
                                     step=epoch * train_data.total_batch_size + batch)

if __name__ == "__main__":
    main()