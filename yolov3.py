

import os
import numpy as np
import tensorflow as tf
from darknet import DarkNet
from loss import loss
from data_ops import transform_targets
from generate_coco_data import CoCoDataGenrator
from visual_ops import draw_bounding_box
from generate_yolo_tfrecord_files import parse_yolo_coco_tfrecord

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class YoloV3:

    def __init__(self,
                 classes,
                 num_class,
                 image_shape=[640, 640, 3],
                 is_training=True,
                 batch_size=5,
                 yolo_max_boxes=100,
                 yolo_iou_threshold=0.5,
                 yolo_score_threshold=0.5):
        self.classes = classes
        self.image_shape = image_shape
        self.is_training = is_training
        self.batch_size = batch_size
        self.yolo_max_boxes = yolo_max_boxes
        self.yolo_iou_threshold = yolo_iou_threshold
        self.yolo_score_threshold = yolo_score_threshold

        self.num_class = num_class
        self.anchors = np.array([[17, 20], [43, 52], [66, 127], [132, 69], [116, 243], [205, 149],
                                 [233, 363], [410, 216], [496, 440]], np.float32) / self.image_shape[0]
        # self.anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
        #                          (59, 119), (116, 90), (156, 198), (373, 326)],
        #                         np.float32) / self.image_shape[0]
        self.anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        self.darnet = DarkNet()
        self.yolo_model = self.build_graph(is_training=self.is_training)

    def yolo_head(self, feature_maps, filters, num_anchors, num_class):
        if isinstance(feature_maps, tuple):
            x, x_skip = feature_maps[0], feature_maps[1]

            # concat with skip connection
            x = self.darnet._darknet_conv(x, filters, 1)
            x = tf.keras.layers.UpSampling2D(2)(x)
            x = tf.keras.layers.Concatenate()([x, x_skip])
        else:
            x = feature_maps

        x = self.darnet._darknet_conv(x, filters, 1)
        x = self.darnet._darknet_conv(x, filters * 2, 3)
        x = self.darnet._darknet_conv(x, filters, 1)
        x = self.darnet._darknet_conv(x, filters * 2, 3)
        x = self.darnet._darknet_conv(x, filters, 1)
        concat_output = x

        x = self.darnet._darknet_conv(x, filters * 2, 3)
        # [batch, h, w, num_anchors * (num_class + 5)]
        x = self.darnet._darknet_conv(x, num_anchors * (num_class + 5), 1, batch_norm=False)
        # [batch, h, w, num_anchors, (num_class + 5)]
        x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                                            num_anchors, num_class + 5)))(x)
        return concat_output, x

    def yolo_boxes(self, pred, anchors, num_classes):
        """ 最后的预测结果

        :param pred: [batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes)]
        :param anchors: [[,],[,],[,]]
        :param classes:
        :return: bbox: [batch_size, grid, grid, anchors, (x1, y1, x2, y2)]
                 objectness: [batch_size, grid, grid, anchors, 1]
                 class_probs: [batch_size, grid, grid, anchors, num_classes]
                 pred_box: [batch_size, grid, grid, anchors, (tx, ty, tw, th)]
        """
        # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        grid_size = tf.shape(pred)[1:3]
        box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, num_classes), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        # class_probs = tf.sigmoid(class_probs)
        class_probs = tf.keras.layers.Softmax()(class_probs)
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        # grid = _meshgrid(grid_size[1],grid_size[0])
        grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        # 这里xy做了归一化, anchors在传进来前也做了归一化
        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2

        x1, y1 = tf.split(box_x1y1, (1, 1), axis=-1)
        x2, y2 = tf.split(box_x2y2, (1, 1), axis=-1)

        x1 = tf.minimum(tf.maximum(x1, 0.), self.image_shape[1])
        y1 = tf.minimum(tf.maximum(y1, 0.), self.image_shape[0])
        x2 = tf.minimum(tf.maximum(x2, 0.), self.image_shape[1])
        y2 = tf.minimum(tf.maximum(y2, 0.), self.image_shape[0])

        bbox = tf.concat([x1, y1, x2, y2], axis=-1)

        return bbox, objectness, class_probs, pred_box

    def yolo_nms(self, yolo_pred, num_class):
        """ 对边框做非极大抑制

        :param yolo_pred: ([boxes, objectness, class_probs],
                           [boxes, objectness, class_probs],
                           [boxes, objectness, class_probs])

               boxes: [batch_size, grid, grid, anchors, (x1, y1, x2, y2)]
               objectness: [batch_size, grid, grid, anchors, 1]
               class_probs: [batch_size, grid, grid, anchors, num_classes]
        :param num_class:
        :return: boxes: [1, nms_nums, (x1, y1, x2, y2)]
                 scores: [1, nms_nums]
                 scores: [1, nms_nums]
                 num_valid_nms_boxes: [1, 1]
        """
        boxes, objectness, class_probs = [], [], []

        # pred: [bbox, objectness, class_probs]
        for pred in yolo_pred:
            # boxes: [batch, -1, 4]
            boxes.append(tf.reshape(pred[0], (tf.shape(pred[0])[0], -1, tf.shape(pred[0])[-1])))
            # objectness: [batch, -1, 1]
            objectness.append(tf.reshape(pred[1], (tf.shape(pred[1])[0], -1, tf.shape(pred[1])[-1])))
            # class_probs: [batch, -1, num_classes]
            class_probs.append(tf.reshape(pred[2], (tf.shape(pred[2])[0], -1, tf.shape(pred[2])[-1])))

        # 这里concat在axis=1
        bbox = tf.concat(boxes, axis=1)
        objectness = tf.concat(objectness, axis=1)
        class_probs = tf.concat(class_probs, axis=1)

        final_batch_nms_bboxes = []
        final_batch_nms_scores = []
        final_batch_nms_classes = []
        valid_detection_nums = []
        for b in range(self.batch_size):
            # 目标概率*类别概率作为最终nms的排序依据
            cur_scores = objectness[b] * class_probs[b]

            # test模式下，batch纬度都是1了, 源码是直接squeeze因为test的batch=1
            # dscores = tf.squeeze(scores, axis=0)
            cur_dscores = tf.reshape(cur_scores, (-1, num_class))
            cur_bbox = tf.reshape(bbox[b], (-1, 4))

            for i in range(num_class):
                cur_dscores_cls = cur_dscores[:, i]

            # 取所有类别中概率最大的
            cur_scores = tf.reduce_max(cur_dscores, [1])
            cur_classes = tf.argmax(cur_dscores, 1)
            selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
                boxes=cur_bbox,
                scores=cur_scores,
                max_output_size=self.yolo_max_boxes,
                iou_threshold=self.yolo_iou_threshold,
                score_threshold=self.yolo_score_threshold,
                soft_nms_sigma=0.5
            )

            # num_valid_nms_boxes = tf.shape(selected_indices)[0]
            # pad_num = self.yolo_max_boxes - num_valid_nms_boxes
            # 数量不够的话做padding
            # selected_indices = tf.concat([selected_indices, tf.zeros(self.yolo_max_boxes - num_valid_nms_boxes, tf.int32)],
            #                              0)
            # selected_scores = tf.concat([selected_scores, tf.zeros(self.yolo_max_boxes - num_valid_nms_boxes, tf.float32)],
            #                             -1)

            vaild_num = tf.shape(selected_indices)[0]
            valid_detection_nums.append(vaild_num)
            pad_num = self.yolo_max_boxes - vaild_num

            # [N, (x1, y1, x2, y2)]
            cur_bbox = tf.gather(cur_bbox, selected_indices)
            cur_bbox = tf.pad(cur_bbox, [[0, pad_num], [0, 0]])
            cur_bbox = tf.expand_dims(cur_bbox, axis=0)
            final_batch_nms_bboxes.append(cur_bbox)

            # [1, N]
            cur_scores = selected_scores
            cur_scores = tf.pad(cur_scores, [[0, pad_num]])
            cur_scores = tf.expand_dims(cur_scores, axis=0)
            final_batch_nms_scores.append(cur_scores)

            # [1, N]
            cur_classes = tf.gather(cur_classes, selected_indices)
            cur_classes = tf.pad(cur_classes, [[0, pad_num]])
            cur_classes = tf.expand_dims(cur_classes, axis=0)
            final_batch_nms_classes.append(cur_classes)

        final_batch_nms_bboxes = tf.concat(final_batch_nms_bboxes, axis=0)
        final_batch_nms_scores = tf.concat(final_batch_nms_scores, axis=0)
        final_batch_nms_classes = tf.concat(final_batch_nms_classes, axis=0)
        # valid_detections = num_valid_nms_boxes
        # valid_detections = tf.expand_dims(valid_detections, axis=0)

        return final_batch_nms_bboxes, final_batch_nms_scores, final_batch_nms_classes, valid_detection_nums
        # return boxes, scores, classes, valid_detections

    def build_graph(self, is_training=True):
        inputs = tf.keras.layers.Input(shape=self.image_shape, name='input_images')

        # [1/8, 1/16, 1/32]
        x1, x2, x3 = self.darnet.build_darknet(inputs, "darknet")

        # 三层预测输出, 第一层, 1/32
        x3, first_out = self.yolo_head(feature_maps=x3,
                                       filters=512,
                                       num_anchors=len(self.anchor_masks[0]),
                                       num_class=self.num_class)
        first_out_bbox, first_out_objectness, first_out_class_probs, first_out_pred_box = tf.keras.layers.Lambda(
            lambda x: self.yolo_boxes(x, self.anchors[self.anchor_masks[0]], self.num_class),
            name='yolo_boxes_first_out')(first_out)

        # 三层预测输出, 第二层, 1/16
        x3, second_out = self.yolo_head(feature_maps=(x3, x2),
                                        filters=256,
                                        num_anchors=len(self.anchor_masks[1]),
                                        num_class=self.num_class)
        second_out_bbox, second_out_objectness, second_out_class_probs, second_out_pred_box = tf.keras.layers.Lambda(
            lambda x: self.yolo_boxes(x, self.anchors[self.anchor_masks[1]], self.num_class),
            name='yolo_boxes_second_out')(second_out)

        # 三层预测输出, 第三层, 1/8
        x3, third_out = self.yolo_head(feature_maps=(x3, x1),
                                       filters=128,
                                       num_anchors=len(self.anchor_masks[2]),
                                       num_class=self.num_class)
        third_out_bbox, third_out_objectness, third_out_class_probs, third_out_pred_box = tf.keras.layers.Lambda(
            lambda x: self.yolo_boxes(x, self.anchors[self.anchor_masks[2]], self.num_class),
            name='yolo_boxes_third_out')(third_out)

        if is_training:
            return tf.keras.models.Model(inputs=inputs, outputs=[
                [first_out_bbox, first_out_objectness, first_out_class_probs, first_out_pred_box],
                [second_out_bbox, second_out_objectness, second_out_class_probs, second_out_pred_box],
                [third_out_bbox, third_out_objectness, third_out_class_probs, third_out_pred_box]
            ])

        # boxes_0 = tf.keras.layers.Lambda(
        #     lambda x: self.yolo_boxes(x, self.anchors[self.anchor_masks[0]], self.num_class),
        #     name='yolo_boxes_0')(first_out)
        # boxes_1 = tf.keras.layers.Lambda(
        #     lambda x: self.yolo_boxes(x, self.anchors[self.anchor_masks[1]], self.num_class),
        #     name='yolo_boxes_1')(second_out)
        # boxes_2 = tf.keras.layers.Lambda(
        #     lambda x: self.yolo_boxes(x, self.anchors[self.anchor_masks[2]], self.num_class),
        #     name='yolo_boxes_2')(third_out)

        outputs = tf.keras.layers.Lambda(lambda x: self.yolo_nms(x, self.num_class),
                                         name='yolo_nms')((
            # boxes_0[:3], boxes_1[:3], boxes_2[:3]
            [first_out_bbox, first_out_objectness, first_out_class_probs],
            [second_out_bbox, second_out_objectness, second_out_class_probs],
            [third_out_bbox, third_out_objectness, third_out_class_probs]
        ))
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def train(self, epochs, log_dir):

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        train_data = CoCoDataGenrator(
            coco_annotation_file="../../data/coco2017/annotations/instances_val2017.json",
            img_shape=self.image_shape,
            batch_size=self.batch_size,
            max_instances=100
        )
        self.classes = train_data.coco.cats
        summary_writer = tf.summary.create_file_writer(log_dir)

        for epoch in range(epochs):
            for batch in range(train_data.total_batch_size):
                with tf.GradientTape() as tape:
                    data = train_data.next_batch()
                    gt_imgs = data['imgs'] / 255.
                    gt_boxes = data['bboxes'] / self.image_shape[0]
                    gt_classes = data['labels']

                    yolo_targets = transform_targets(
                        gt_boxes=gt_boxes,
                        gt_lables=gt_classes,
                        anchors=self.anchors,
                        anchor_masks=self.anchor_masks,
                        im_size=self.image_shape[0]
                    )
                    yolo_preds = self.yolo_model(gt_imgs, training=True)

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
                            anchors=self.anchors[self.anchor_masks[i]],
                            ignore_thresh=0.5
                        )
                        # print(i, tf.reduce_mean(xy_loss),  tf.reduce_mean(obj_loss))

                        total_xy_loss += tf.reduce_mean(xy_loss)
                        total_wh_loss += tf.reduce_mean(wh_loss)
                        total_obj_loss += tf.reduce_mean(obj_loss)
                        total_class_loss += tf.reduce_mean(class_loss)

                    total_loss = total_xy_loss + total_wh_loss + total_obj_loss + total_class_loss
                    grad = tape.gradient(total_loss, self.yolo_model.trainable_variables)
                    optimizer.apply_gradients(zip(grad, self.yolo_model.trainable_variables))

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
                    gt_boxes = gt_boxes[0] * self.image_shape[0]
                    gt_classes = gt_classes[0]
                    non_zero_ids = np.where(np.sum(gt_boxes, axis=-1))[0]
                    for i in non_zero_ids:
                        label = gt_classes[i]
                        class_name = self.classes[label]['name']
                        xmin, ymin, xmax, ymax = gt_boxes[i]
                        gt_img = draw_bounding_box(gt_img, class_name, label, int(xmin), int(ymin), int(xmax),
                                                   int(ymax))

                    # pred
                    pred_img = gt_imgs[0].copy() * 255
                    boxes, scores, classes, valid_detection_nums = self.yolo_nms(yolo_preds, self.num_class)
                    # print(scores)
                    # print(gt_classes)
                    for i in range(valid_detection_nums[0]):
                        if scores[0][i] > 0.5:
                            label = classes[0][i].numpy()
                            if self.classes.get(label):
                                class_name = self.classes[label]['name']
                                xmin, ymin, xmax, ymax = boxes[0][i] * self.image_shape[0]
                                pred_img = draw_bounding_box(pred_img, class_name, scores[0][i], int(xmin), int(ymin),
                                                             int(xmax), int(ymax))

                    concat_imgs = tf.concat([gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1)
                    summ_imgs = tf.expand_dims(concat_imgs, 0)
                    summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                    with summary_writer.as_default():
                        tf.summary.image("imgs/gt,pred,epoch{}".format(epoch), summ_imgs,
                                         step=epoch * train_data.total_batch_size + batch)
                        # tf.summary.image("imgs/gt,pred,epoch{}".format(step // 1500), summ_imgs, step=step)

    def train_with_tfrecord(self, epochs, log_dir):

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        train_data = CoCoDataGenrator(
            coco_annotation_file="../../data/coco2017/annotations/instances_val2017.json",
            img_shape=self.image_shape,
            batch_size=self.batch_size,
            max_instances=100
        )
        self.classes = train_data.coco.cats
        train_data.total_batch_size = 750
        train_tfrecord_data = parse_yolo_coco_tfrecord(
            is_training=True,
            tfrec_path="../../data/coco2017",
            repeat=1,
            batch=2
        )

        summary_writer = tf.summary.create_file_writer(log_dir)

        for epoch in range(epochs):
            batch = 0
            for gt_imgs, gt_boxes, gt_classes, yolo_targets_0, yolo_targets_1, yolo_targets_2 in train_tfrecord_data:
                batch += 1

                with tf.GradientTape() as tape:
                    yolo_preds = self.yolo_model(gt_imgs, training=True)
                    # 3层输出分别计算损失
                    yolo_targets = [yolo_targets_0, yolo_targets_1, yolo_targets_2]
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
                            anchors=self.anchors[self.anchor_masks[i]],
                            ignore_thresh=0.5
                        )

                        total_xy_loss += tf.reduce_mean(xy_loss)
                        total_wh_loss += tf.reduce_mean(wh_loss)
                        total_obj_loss += tf.reduce_mean(obj_loss)
                        total_class_loss += tf.reduce_mean(class_loss)
                    total_loss = total_xy_loss + total_wh_loss + total_obj_loss + total_class_loss
                    grad = tape.gradient(total_loss, self.yolo_model.trainable_variables)
                    optimizer.apply_gradients(zip(grad, self.yolo_model.trainable_variables))

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
                    gt_imgs = np.array(gt_imgs, np.float64)
                    gt_boxes = np.array(gt_boxes, np.float64)
                    gt_classes = np.array(gt_classes, np.int8)

                    gt_img = gt_imgs[0].copy() * 255
                    gt_boxes = gt_boxes[0] * self.image_shape[0]
                    gt_classes = gt_classes[0]
                    non_zero_ids = np.where(np.sum(gt_boxes, axis=-1))[0]
                    for i in non_zero_ids:
                        label = gt_classes[i]
                        class_name = self.classes[label]['name']
                        xmin, ymin, xmax, ymax = gt_boxes[i]
                        gt_img = draw_bounding_box(gt_img, class_name, label, int(xmin), int(ymin), int(xmax),
                                                   int(ymax))

                    # pred
                    pred_img = gt_imgs[0].copy() * 255
                    boxes, scores, classes, valid_detection_nums = self.yolo_nms(yolo_preds, self.num_class)
                    for i in range(valid_detection_nums[0]):
                        if scores[0][i] > 0.85:
                            label = classes[0][i].numpy()
                            if self.classes.get(label):
                                class_name = self.classes[label]['name']
                                xmin, ymin, xmax, ymax = boxes[0][i] * self.image_shape[0]
                                pred_img = draw_bounding_box(pred_img, class_name, scores[0][i], int(xmin), int(ymin),
                                                             int(xmax), int(ymax))

                    concat_imgs = tf.concat([gt_img, pred_img], axis=1)
                    summ_imgs = tf.expand_dims(concat_imgs, 0)
                    summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                    with summary_writer.as_default():
                        tf.summary.image("imgs/gt,pred,epoch{}".format(epoch), summ_imgs,
                                         step=epoch * train_data.total_batch_size + batch)


if __name__ == "__main__":
    yolo3 = YoloV3(classes=[], num_class=91, batch_size=1, is_training=True)
    # yolo3.train(101, log_dir='./logs')
    yolo3.train_with_tfrecord(101, log_dir='./logs')

    # yolo3 = YoloV3(classes=["a", 'b', 'c'],num_class=3, batch_size=5)
    # yolo3_model = yolo3.build_graph(is_training=False)
    # yolo3_model.summary(line_length=500)

    # for name,x in zip(yolo3_model.input_names,yolo3_model.inputs):
    #     print(tf.split(x, 4))
    #
    # from tensorflow.python.ops import summary_ops_v2
    # from tensorflow.python.keras.backend import get_graph
    #
    # tb_writer = tf.summary.create_file_writer('./logs')
    # with tb_writer.as_default():
    #     if not yolo3_model.run_eagerly:
    #         summary_ops_v2.graph(get_graph(), step=0)
