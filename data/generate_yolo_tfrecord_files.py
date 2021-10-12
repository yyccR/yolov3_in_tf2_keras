import tensorflow as tf
import numpy as np
from data_ops import transform_targets
from data.generate_coco_data import CoCoDataGenrator


def tensor_feature(value):
    """ tensor序列化成feature
    :param value:
    :return:
    """
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def generate_yolo_coco_tfrecord(is_training=True, tfrec_path='./coco2017/', image_shape=(640, 640, 3),
                                sample_nums=1500, max_instances=100):
    if not os.path.exists(tfrec_path):
        os.mkdir(tfrec_path)

    if is_training:
        coco_file = "./coco2017/annotations/instances_val2017.json"

    coco = CoCoDataGenrator(
        coco_annotation_file=coco_file,
        img_shape=image_shape,
        batch_size=1,
        max_instances=max_instances,
        include_crowd=False,
        include_mask=False,
        include_keypoint=False
    )

    if is_training:
        tfrec_file = os.path.join(tfrec_path, "yolo_coco_train.tfrec")
    else:
        tfrec_file = os.path.join(tfrec_path, "yolo_coco_test.tfrec")
    tfrec_writer = tf.io.TFRecordWriter(tfrec_file)

    anchors = np.array([[17, 20], [43, 52], [66, 127], [132, 69], [116, 243], [205, 149],
                        [233, 363], [410, 216], [496, 440]], np.float32) / image_shape[0]
    anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    for i in range(coco.total_batch_size)[:sample_nums]:
        print("current {} total {}".format(i, coco.total_batch_size))
        # {"img":, "bboxes":, "labels":, "masks":, "key_points":}
        data = coco.next_batch()

        gt_imgs = data['imgs'] / 255.
        gt_boxes = data['bboxes'] / image_shape[0]
        gt_classes = data['labels']

        yolo_targets = transform_targets(
            gt_boxes=gt_boxes,
            gt_lables=gt_classes,
            anchors=anchors,
            anchor_masks=anchor_masks,
            im_size=image_shape[0]
        )

        feature = {
            "image": tensor_feature(gt_imgs),
            "gt_boxes": tensor_feature(gt_boxes),
            "gt_classes": tensor_feature(gt_classes),
            "yolo_targets_0": tensor_feature(yolo_targets[0]),
            "yolo_targets_1": tensor_feature(yolo_targets[1]),
            "yolo_targets_2": tensor_feature(yolo_targets[2]),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        tfrec_writer.write(example.SerializeToString())
    tfrec_writer.close()


def parse_single_example(single_record):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "gt_boxes": tf.io.FixedLenFeature([], tf.string),
        "gt_classes": tf.io.FixedLenFeature([], tf.string),
        "yolo_targets_0": tf.io.FixedLenFeature([], tf.string),
        "yolo_targets_1": tf.io.FixedLenFeature([], tf.string),
        "yolo_targets_2": tf.io.FixedLenFeature([], tf.string),
    }
    feature = tf.io.parse_single_example(single_record, feature_description)

    image = tf.io.parse_tensor(feature['image'], tf.float64)[0]
    gt_boxes = tf.io.parse_tensor(feature['gt_boxes'], tf.float64)[0]
    gt_classes = tf.io.parse_tensor(feature['gt_classes'], tf.int8)[0]

    yolo_targets_0 = tf.io.parse_tensor(feature['yolo_targets_0'], tf.float32)[0]
    yolo_targets_1 = tf.io.parse_tensor(feature['yolo_targets_1'], tf.float32)[0]
    yolo_targets_2 = tf.io.parse_tensor(feature['yolo_targets_2'], tf.float32)[0]

    return image, gt_boxes, gt_classes, yolo_targets_0, yolo_targets_1, yolo_targets_2


def parse_yolo_coco_tfrecord(is_training=True, tfrec_path='./coco2017', repeat=1, shuffle_buffer=1000, batch=2):
    if is_training:
        tfrec_file = os.path.join(tfrec_path, "yolo_coco_train.tfrec")
    else:
        tfrec_file = os.path.join(tfrec_path, "yolo_coco_test.tfrec")

    voc_tfrec_dataset = tf.data.TFRecordDataset(tfrec_file, num_parallel_reads=2)
    parse_data = voc_tfrec_dataset \
        .repeat(repeat) \
        .shuffle(shuffle_buffer) \
        .map(parse_single_example) \
        .batch(batch) \
        .prefetch(10)

    return parse_data


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # generate_yolo_coco_tfrecord(is_training=True)
    parse_data = parse_yolo_coco_tfrecord(is_training=True, repeat=1)
    i = 0
    for _ in range(10):
        for image, gt_boxes, gt_classes, yolo_targets_0, yolo_targets_1, yolo_targets_2 in parse_data:
            print(np.shape(image))
            print(np.shape(gt_boxes))
            print(np.shape(gt_classes))
            print(np.shape(yolo_targets_0))
            print(np.shape(yolo_targets_1))
            print(np.shape(yolo_targets_2))
            i += 1
            print(i)
