import cv2
import numpy as np
import tensorflow as tf


def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    """ 生成YOLO某一层output的目标值
    :param y_true: [N, boxes, (x1, y1, x2, y2, class, best_anchor)]
    :param grid_size:
    :param anchor_idxs: [,,]
    :return: y_true_out: [N, grid, grid, anchors, [x1, y1, x2, y2, obj, class]]
    """
    # y_true: [N, boxes, (x1, y1, x2, y2, class, best_anchor)]
    # print(y_true)
    N, num_boxes, _ = np.shape(y_true)

    # y_true_out: [N, grid, grid, anchors, [x1, y1, x2, y2, obj, class]]
    y_true_out = np.zeros((N, grid_size, grid_size, np.shape(anchor_idxs)[0], 6),dtype=np.float32)

    anchor_idxs = np.array(anchor_idxs, np.int32)
    # indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    # updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    for i in np.arange(N):
        for j in np.arange(num_boxes):
            # 这里如果是padding的数据则跳过
            if y_true[i][j][2] == 0:
                continue
            # print(y_true[i][j][5])
            # 判断跟传进来的anchor idx哪个一样, y_true[i][j][5]为9个best anchor中的某一个
            anchor_eq = anchor_idxs == y_true[i][j][5]
            # print(anchor_eq)

            # 存在一个一样
            if np.any(anchor_eq):
                box = y_true[i][j][0:4]
                # 计算中心点
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                anchor_idx = np.array(np.where(anchor_eq)[0], np.int32)
                grid_xy = np.array(box_xy // (1 / grid_size), np.int32)

                y_true_out[i, grid_xy[1], grid_xy[0], anchor_idx[0], :] = \
                    [box[0], box[1], box[2], box[3], 1, y_true[i,j,4]]
                # print([box[0], box[1], box[2], box[3], 1, y_true[i,j,4]])
                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                # indexes = indexes.write(
                #     idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                # updates = updates.write(
                #     idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])

    # tf.print(indexes.stack())
    # tf.print(updates.stack())
    return y_true_out


def transform_targets(gt_boxes, gt_lables, anchors, anchor_masks, im_size):
    """ 计算
    :param gt_boxes: [batch, num_boxes, (x1, y1, x2, y2)]
    :param gt_lables: [batch, num_boxes]
    :param anchors: [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                    (59, 119), (116, 90), (156, 198), (373, 326)] / im_size
    :param anchor_masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    :param im_size:
    :return:  ([N, grid, grid, anchors, [x1, y1, x2, y2, obj, class]], [], [])
    """
    y_outs = []
    grid_size = im_size // 32

    # 计算anchor的面积, 这里anchor都已经归一化
    anchors = np.array(anchors, np.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]

    # 计算gt_box的宽高, 这里宽高也已经归一化
    box_wh = gt_boxes[..., 2:4] - gt_boxes[..., 0:2]
    box_wh = np.tile(np.expand_dims(box_wh, axis=-2),
                     (1, 1, np.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]

    # 计算iou
    intersection = np.minimum(box_wh[..., 0], anchors[..., 0]) * np.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = np.array(np.argmax(iou, axis=-1), np.float32)
    anchor_idx = np.expand_dims(anchor_idx, axis=-1)
    gt_labels = np.expand_dims(gt_lables, axis=-1)

    # 拼接最后的结果
    y_train = np.concatenate([gt_boxes, gt_labels, anchor_idx], axis=-1)
    # print(y_train)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


if __name__ == "__main__":
    gt_boxes = np.abs(np.random.random([2, 4, 4]))
    labels = np.round(np.random.random([2,4]))
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)], np.int32) / 640.
    anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    im_size = 640
    output = transform_targets(
        gt_boxes=gt_boxes,
        gt_lables=labels,
        anchors=anchors,
        anchor_masks=anchor_masks,
        im_size=im_size
    )
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)

    # box_labels = np.concatenate([gt_boxes, np.expand_dims(labels, axis=-1)], axis=-1)
    # box_labels = np.array(box_labels, dtype=np.float32)
    # output2 = transform_targets2(
    #     y_train=box_labels,
    #     anchors=anchors,
    #     anchor_masks=anchor_masks,
    #     size=im_size
    # )
    #
    # print(np.sum(output2[0].numpy() - output[0]))
    # print(np.sum(output2[1].numpy() - output[1]))
    # print(np.sum(output2[2].numpy() - output[2]))
