import sys
import cv2
import os
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io


class CoCoDataGenrator:
    def __init__(self,
                 coco_annotation_file,
                 img_shape=(640, 640, 3),
                 batch_size=1,
                 max_instances=100,
                 include_crowd=False,
                 include_mask=False,
                 include_keypoint=False,
                 image_mean=np.array([[[102.9801, 115.9465, 122.7717]]]),
                 use_mini_mask=True,
                 mini_mask_shape=(56, 56),
                 data_size = -1
                 ):
        self.coco_annotation_file = coco_annotation_file
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.max_instances = max_instances
        self.include_crowd = include_crowd
        self.include_mask = include_mask
        self.include_keypoint = include_keypoint
        self.img_mean = image_mean
        self.use_mini_mask = use_mini_mask
        self.mini_mask_shape = mini_mask_shape
        self.data_size = data_size

        self.current_batch_index = 0
        self.total_batch_size = 0
        self.img_ids = []
        self.coco = COCO(annotation_file=coco_annotation_file)
        self.load_data()

    def load_data(self):
        # 初步过滤数据是否包含crowd
        target_img_ids = []
        for k in self.coco.imgToAnns:
            annos = self.coco.imgToAnns[k]
            if annos:
                annos = list(filter(lambda x: x['iscrowd'] == self.include_crowd, annos))
                if annos:
                    target_img_ids.append(k)
        if self.data_size >= 0:
            target_img_ids = target_img_ids[:self.data_size]
        self.total_batch_size = len(target_img_ids) // self.batch_size
        self.img_ids = target_img_ids

    def next_batch(self):
        if self.current_batch_index >= self.total_batch_size:
            self.current_batch_index = 0
            self._on_epoch_end()

        batch_img_ids = self.img_ids[self.current_batch_index * self.batch_size:
                                     (self.current_batch_index + 1) * self.batch_size]
        batch_imgs = []
        batch_bboxes = []
        batch_labels = []
        batch_masks = []
        batch_keypoints = []
        valid_nums = []
        for img_id in batch_img_ids:
            # {"img":, "bboxes":, "labels":, "masks":, "key_points":}
            data = self._data_generation(image_id=img_id)
            if len(np.shape(data['img'])) > 0:
                batch_imgs.append(data['img'])
                batch_labels.append(data['labels'])
                batch_bboxes.append(data['bboxes'])
                valid_nums.append(data['valid_nums'])
                # if len(data['labels']) > self.max_instances:
                #     batch_bboxes.append(data['bboxes'][:self.max_instances, :])
                #     batch_labels.append(data['labels'][:self.max_instances])
                #     valid_nums.append(self.max_instances)
                # else:
                #     pad_num = self.max_instances - len(data['labels'])
                #     batch_bboxes.append(np.pad(data['bboxes'], [(0, pad_num), (0, 0)]))
                #     batch_labels.append(np.pad(data['labels'], [(0, pad_num)]))
                #     valid_nums.append(len(data['labels']))

                if self.include_mask:
                    batch_masks.append(data['masks'])

                if self.include_keypoint:
                    batch_keypoints.append(data['keypoints'])

        self.current_batch_index += 1
        if len(batch_imgs) < self.batch_size:
            return self.next_batch()

        output = {
            'imgs': np.array(batch_imgs, dtype=np.float32),
            'bboxes': np.array(batch_bboxes, dtype=np.float32),
            'labels': np.array(batch_labels, dtype=np.int8),
            'masks': np.array(batch_masks, dtype=np.int8),
            'keypoints': np.array(batch_keypoints, dtype=np.float32),
            'valid_nums': np.array(valid_nums, dtype=np.int8)
        }

        return output

    def _on_epoch_end(self):
        np.random.shuffle(self.img_ids)

    def _resize_im(self, origin_im, bboxes):
        """ 对图片/mask/box resize

        :param origin_im
        :param bboxes
        :return im_blob: [h, w, 3]
                gt_boxes: [N, [ymin, xmin, ymax, xmax]]
        """
        im_shape = np.shape(origin_im)
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.img_shape[0]) / float(im_size_max)

        # resize原始图片
        im_resize = cv2.resize(origin_im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_resize_shape = np.shape(im_resize)
        im_blob = np.zeros(self.img_shape, dtype=np.float32)
        im_blob[0:im_resize_shape[0], 0:im_resize_shape[1], :] = im_resize

        # resize对应边框
        bboxes_resize = np.array(bboxes * im_scale, dtype=np.float32)

        return im_blob, bboxes_resize

    def _resise_mini_mask(self, masks, boxes):
        """  mask处理成最小mask """
        mini_masks = []
        h, w, c = np.shape(masks)
        for i in range(c):
            ymin, xmin, ymax, xmax = boxes[i]
            mask = masks[int(ymin):int(ymax), int(xmin):int(xmax), i]
            mini_m = cv2.resize(mask, self.mini_mask_shape, interpolation=cv2.INTER_LINEAR)
            mini_m = np.array(mini_m >= 0.5, dtype=np.int8)
            mini_m = np.expand_dims(mini_m, axis=-1)
            mini_masks.append(mini_m)
        mini_masks = np.concatenate(mini_masks, axis=-1)
        return mini_masks

    def _resize_mask(self, origin_masks):
        """ resize mask数据
        :param origin_mask:
        :return: mask_resize: [h, w, instance]
                 gt_boxes: [N, [ymin, xmin, ymax, xmax]]
        """
        mask_shape = np.shape(origin_masks)
        mask_size_max = np.max(mask_shape[0:3])
        im_scale = float(self.img_shape[0]) / float(mask_size_max)

        # resize mask/box
        gt_boxes = []
        masks_resize = []
        for m in origin_masks:
            m = np.array(m, dtype=np.float32)
            m_resize = cv2.resize(m, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            m_resize = np.array(m_resize >= 0.5, dtype=np.int8)

            # 计算bdbox
            h, w = np.shape(m_resize)
            rows, cols = np.where(m_resize)
            # [xmin, ymin, xmax, ymax]
            xmin = np.min(cols) if np.min(cols) >= 0 else 0
            ymin = np.min(rows) if np.min(rows) >= 0 else 0
            xmax = np.max(cols) if np.max(cols) <= w else w
            ymax = np.max(rows) if np.max(rows) <= h else h
            bdbox = [ymin, xmin, ymax, xmax]
            gt_boxes.append(bdbox)

            mask_blob = np.zeros((self.img_shape[0], self.img_shape[1], 1), dtype=np.float32)
            mask_blob[0:h, 0:w, 0] = m_resize
            masks_resize.append(mask_blob)

        # [instance_num, [xmin, ymin, xmax, ymax]]
        gt_boxes = np.array(gt_boxes, dtype=np.int16)
        # [h, w, instance_num]
        masks_resize = np.concatenate(masks_resize, axis=-1)

        return masks_resize, gt_boxes

    def _data_generation(self, image_id):
        """ 拉取coco标记数据, 目标边框, 类别, mask
        :param image_id:
        :return:
        """

        anno_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=self.include_crowd)
        bboxes = []
        labels = []
        masks = []
        keypoints = []

        for i in anno_ids:
            # 边框, 处理成左上右下坐标
            ann = self.coco.anns[i]
            bbox = ann['bbox']
            xmin, ymin, w, h = bbox
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            bboxes.append([ymin, xmin, ymax, xmax])
            # 类别ID
            label = ann['category_id']
            labels.append(label)
            # 实例分割
            if self.include_mask:
                # [instances, h, w]
                mask = self.coco.annToMask(ann)
                # cv2.imshow("mask", np.array(mask,dtype=np.uint8)*255)
                # cv2.imshow("img", img)
                # cv2.waitKey(0)
                masks.append(mask)
            if self.include_keypoint and ann.get('keypoints'):
                keypoint = ann['keypoints']
                # 处理成[x,y,v] 其中v=0表示没有此点,v=1表示被挡不可见,v=2表示可见
                keypoint = np.reshape(keypoint, [-1, 3])
                keypoints.append(keypoint)

        # 输出包含5个东西, 不需要则为空
        outputs = {
            "img": [],
            "labels": [],
            "bboxes": [],
            "masks": [],
            "keypoints": [],
            "valid_nums": 0
        }

        valid_nums = 0
        if len(labels) > self.max_instances:
            bboxes = bboxes[:self.max_instances, :]
            labels = labels[:self.max_instances]
            valid_nums = self.max_instances
            # batch_bboxes.append(data['bboxes'][:self.max_instances, :])
            # batch_labels.append(data['labels'][:self.max_instances])
            # valid_nums.append(self.max_instances)
        else:
            pad_num = self.max_instances - len(labels)
            bboxes = np.pad(bboxes, [(0, pad_num), (0, 0)])
            labels = np.pad(labels, [(0, pad_num)])
            valid_nums = self.max_instances - pad_num
            # batch_bboxes.append(np.pad(data['bboxes'], [(0, pad_num), (0, 0)]))
            # batch_labels.append(np.pad(data['labels'], [(0, pad_num)]))
            # valid_nums.append(len(data['labels']))

        # 处理最终数据 mask
        if self.include_mask:
            # [h, w, instances]
            masks, mask_boxes = self._resize_mask(origin_masks=masks)
            # mini mask
            if self.use_mini_mask:
                masks = self._resise_mini_mask(masks, mask_boxes)
            if np.shape(masks)[2] > self.max_instances:
                masks = masks[:self.max_instances, :, :]
            else:
                pad_num = self.max_instances - np.shape(masks)[2]
                masks = np.pad(masks, [(0, 0), (0, 0), (0, pad_num)])

            outputs['masks'] = masks
            # outputs['bboxes'] = bboxes

        # 处理最终数据 keypoint
        if self.include_keypoint:
            keypoints = np.array(keypoints, dtype=np.int8)
            outputs['keypoints'] = keypoints

        img_coco_url_file = str(self.coco.imgs[image_id].get('coco_url',""))
        img_url_file = str(self.coco.imgs[image_id].get('url',""))
        img_local_file = str(self.coco.imgs[image_id].get('file_name',""))
        img_local_file = os.path.join(os.path.dirname(self.coco_annotation_file), img_local_file)
        img = []

        if os.path.isfile(img_local_file):
            img = io.imread(img_local_file)
        elif img_coco_url_file.startswith("http"):
            img = io.imread(self.coco.imgs[image_id]['coco_url'])
        elif img_url_file.startswith("http"):
            img = io.imread(self.coco.imgs[image_id]['coco_url'])
        else:
            return outputs
        if len(np.shape(img)) < 2:
            return outputs
        elif len(np.shape(img)) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.pad(img, [(0, 0), (0, 0), (0, 2)])
        else:
            img = img[:, :, ::-1]

        labels = np.array(labels, dtype=np.int8)
        bboxes = np.array(bboxes, dtype=np.float32)
        img_resize, bboxes_resize = self._resize_im(origin_im=img, bboxes=bboxes)

        outputs['img'] = img_resize - self.img_mean
        outputs['labels'] = labels
        outputs['bboxes'] = bboxes_resize
        outputs['valid_nums'] = valid_nums

        return outputs


if __name__ == "__main__":
    file = "./instances_val2017.json"
    coco = CoCoDataGenrator(coco_annotation_file=file,include_mask=True,include_keypoint=True)
    # data = coco.next_batch()
    # print(data)
    # data = coco.next_batch()
    # print(data)
    # for i in coco.coco.cats:
    #     print(coco.coco.cats[i])
    # class_names = list(map(lambda x:x['name'],coco.coco.cats))

    # coco = COCO(annotation_file=file)
    #
    # print("---------------------------")
    # for anno in coco.dataset['info']:
    #     print(anno, coco.dataset['info'][anno])
    #
    # print("---------------------------")
    # for anno in coco.dataset['licenses']:
    #     print(anno)
    #
    # print("---------------------------")
    # for anno in coco.dataset['categories']:
    #     print(anno)
    #
    # print("---------------------------")
    # for anno in coco.dataset['images']:
    #     print(anno)

    # print("---------------------------")
    # for anno in coco.dataset['annotations']:
    #     print(anno)
    # anno = coco.anns[900100259690]