import tensorflow as tf


def broadcast_iou(box_1, box_2):
    """ 计算最终iou

    :param box_1:
    :param box_2:
    :return: [batch_size, grid, grid, anchors, num_gt_box]
    """
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def loss(pred_box, pred_box_xywh, true_box, pred_obj, true_obj, pred_class, true_class, anchors, ignore_thresh,
         balanced_rate=5):
    # def loss(preds, targets, anchors, ignore_thresh)
    """
    :param pred_box: [batch_size, grid, grid, anchors, (x1, y1, x2, y2)]
    :param pred_box_xywh: [batch_size, grid, grid, anchors, (tx, ty, tw, th)]
    :param true_box: [batch_size, grid, grid, anchors, (x1, y1, x2, y2)]
    :param pred_obj: [batch_size, grid, grid, anchors, 1]
    :param true_obj: [batch_size, grid, grid, anchors, 1]
    :param pred_class: [batch_size, grid, grid, anchors, num_classes]
    :param true_class: [batch_size, grid, grid, anchors, 1]
    :param anchors: [[w1,h1],[w2,h2],[w3,h3]]
    :param ignore_thresh: 正负样本iou阈值
    :param balanced_rate: 正负样本平衡比例
    :return:
    """
    # [batch_size, grid, grid, anchors, 2]
    pred_xy = pred_box_xywh[..., 0:2]
    # [batch_size, grid, grid, anchors, 2]
    pred_wh = pred_box_xywh[..., 2:4]

    # true_box, true_obj, true_class_idx = tf.split(true_box, (4, 1, 1), axis=-1)
    true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
    true_wh = true_box[..., 2:4] - true_box[..., 0:2]

    # give higher weights to small boxes
    box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

    # 3. inverting the pred box equations
    grid_size = tf.shape(true_box)[1]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    # [grid_size, grid_size, 1, 2]
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    # 计算true_box的平移缩放量
    # [batch_size, grid, grid, anchors, 2]
    true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
    # [batch_size, grid, grid, anchors, 2]

    true_wh = tf.math.log(true_wh / anchors)
    true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

    # 4. calculate all masks
    # [batch_size, grid, grid, anchors]
    obj_mask = tf.squeeze(true_obj, -1)
    positive_num = tf.cast(tf.reduce_sum(obj_mask), tf.int32) + 1
    negative_num = balanced_rate * positive_num
    # ignore false positive when iou is over threshold
    # [batch_size, grid, grid, anchors, num_gt_box] => [batch_size, grid, grid, anchors, 1]

    best_iou = tf.map_fn(
        lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
            x[1], tf.cast(x[2], tf.bool))), axis=-1),
        (pred_box, true_box, obj_mask),
        tf.float32)
    # [batch_size, grid, grid, anchors, 1]
    ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
    # 这里做了下样本均衡.
    ignore_num = tf.cast(tf.reduce_sum(ignore_mask), tf.int32)
    if ignore_num > negative_num:
        neg_inds = tf.random.shuffle(tf.where(ignore_mask))[:negative_num]
        neg_inds = tf.expand_dims(neg_inds, axis=1)
        ones = tf.ones(tf.shape(neg_inds)[0], tf.float32)
        ones = tf.expand_dims(ones, axis=1)
        # 更新mask
        ignore_mask = tf.zeros_like(ignore_mask, tf.float32)
        ignore_mask = tf.tensor_scatter_nd_add(ignore_mask, neg_inds, ones)

    # 5. calculate all losses
    # [batch_size, grid, grid, anchors]
    xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    # [batch_size, grid, grid, anchors]
    wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

    # obj_loss = binary_crossentropy(true_obj, pred_obj)
    conf_focal = tf.pow(obj_mask - tf.squeeze(pred_obj, -1), 2)
    obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
    obj_loss = conf_focal * (obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss)

    # obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
    # 这里除了正样本会计算损失, 负样本低于一定置信的也计算损失
    # obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

    # TODO: use binary_crossentropy instead
    # class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)
    class_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(true_class, pred_class)

    # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
    xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
    wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
    obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
    class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

    # return xy_loss + wh_loss + obj_loss + class_loss
    return xy_loss, wh_loss, obj_loss, class_loss


if __name__ == "__main__":
    pass
