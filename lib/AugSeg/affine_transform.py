import scipy.ndimage as ndimage
import numpy as np
from PIL import Image
import pycocotools.mask as cocomask
from .pycococreater import binary_mask_to_rle


def __bbox_from_bboxes(bboxes: list):
    """
    Compute the overall bbox of multiple bboxes
    :param bboxes: list of bounding boxes. (should be list of list)
    :return: overall bbox
    """
    bbox_iter = iter(bboxes)
    bbox = next(bbox_iter)
    xmin = bbox[0]
    xmax = bbox[0] + bbox[2]
    ymin = bbox[1]
    ymax = bbox[1] + bbox[3]
    while True:
        try:
            bbox_part = next(bbox_iter)
            _xmin = bbox_part[0]
            _xmax = bbox_part[0] + bbox_part[2]
            _ymin = bbox_part[1]
            _ymax = bbox_part[1] + bbox_part[3]
        except StopIteration:
            break
        xmin = min(xmin, _xmin)
        ymin = min(ymin, _ymin)
        xmax = max(xmax, _xmax)
        ymax = max(ymax, _ymax)
    bbox = [xmin, ymin, (xmax - xmin), (ymax - ymin)]
    return bbox


def __transform_img(img, trans_param, new_shape):
    canvas = np.zeros(new_shape, dtype=np.uint8)
    if len(img.shape) == 3:
        canvas[0:img.shape[0], 0:img.shape[1], 0:img.shape[2]] = img
    elif len(img.shape) == 2:
        canvas[0:img.shape[0], 0:img.shape[1]] = img
    else:
        raise ValueError('image shape not understood {}'.format(img.shape))

    s = trans_param['s']
    theta = trans_param['theta']
    tx = trans_param['tx']
    ty = trans_param['ty']

    # here t is the inverse of widely-known similarity transform matrix,
    # since ndimage.affine_transform is from new image to source
    t = np.array([[1 / s * np.cos(theta), 1 / s * np.sin(theta)],
                  [-1 / s * np.sin(theta), 1 / s * np.cos(theta)]])
    o = [-tx / s * np.cos(theta) - ty / s * np.sin(theta) + img.shape[1] / 2,
         tx / s * np.sin(theta) - ty / s * np.cos(theta) + img.shape[0] / 2]

    if len(canvas.shape) == 3:  # colored image
        chnls = np.dsplit(canvas.transpose((1, 0, 2)), canvas.shape[2])
        canvas = [ndimage.affine_transform(x[:, :, 0], t, o) for x in chnls]
        canvas = np.stack(canvas, axis=2)
        canvas = canvas.transpose((1, 0, 2))
    elif len(canvas.shape) == 2:  # single channel image
        canvas = ndimage.affine_transform(canvas, t, o)
    else:
        raise ValueError('canvas shape not understood {}'.format(canvas.shape))

    return np.asfortranarray(canvas)


def transform_image(bkg_img: np.ndarray, inst_imgs: list, trans_params: list):
    canvas_h, canvas_w = bkg_img.shape[0:2]
    for inst_img, trans_param in zip(inst_imgs, trans_params):
        canvas_d = inst_img.shape[2]

        canvas = __transform_img(inst_img, trans_param, (canvas_h, canvas_w, canvas_d))

        # alpha-composite image
        bkg_img = [x[:, :, 0] * (1 - canvas[:, :, 3] / 255) + y[:, :, 0] * (canvas[:, :, 3] / 255)
                   for x, y in zip(np.dsplit(bkg_img, 3), np.dsplit(canvas[:, :, 0:3], 3))]

        bkg_img = np.stack(bkg_img, axis=2)
        bkg_img = bkg_img.astype(np.uint8)

    return bkg_img


def transform_annotation(anns: list, trans_params: list, group_bnds: list, group_indices: list, width: int, height: int):
    ret_anns = []
    for indices, trans_param, group_bnd in zip(group_indices, trans_params, group_bnds):  #!!not equal length
        for idx in indices:
            ann = anns[idx]
            xmin, ymin, xmax, ymax = group_bnd
            segm = ann['segmentation']
            if type(segm) == list:
                # polygon
                s = trans_param['s']
                theta = trans_param['theta']
                tx = trans_param['tx']
                ty = trans_param['ty']
                trans = np.array([[s * np.cos(theta), -s * np.sin(theta)],
                                  [s * np.sin(theta), s * np.cos(theta)]])
                offset = np.array([tx, ty])

                new_segm = []
                for segm_part in segm:
                    xs = np.array([segm_part[0::2]]) - xmin - (xmax - xmin) / 2
                    ys = np.array([segm_part[1::2]]) - ymin - (ymax - ymin) / 2
                    if 'flip' in trans_param:
                        if trans_param['flip'] == 'horizontal':
                            xs = -xs
                        elif trans_param['flip'] == 'vertical':
                            ys = -ys
                        else:
                            raise ValueError('Unknown flip parameter {}'.format(trans_param['flip']))
                    xys = np.vstack([xs, ys])
                    xys = np.dot(trans, xys)
                    xys = [xy + o for xy, o in zip(xys, offset)]
                    xys = np.vstack(xys)
                    xs = np.clip(xys[0, :], 0, width - 1)
                    ys = np.clip(xys[1, :], 0, height - 1)
                    poly = np.stack((xs, ys), axis=-1)
                    new_segm.append(poly.reshape(-1).tolist())
                rle = cocomask.frPyObjects(new_segm, height, width)
                bboxes = cocomask.toBbox(rle).tolist()
                bbox = __bbox_from_bboxes(bboxes)
            elif type(segm['counts']) == list:
                # uncompressed RLE
                rle = cocomask.frPyObjects(segm, height, width)
                inst_mask = cocomask.decode(rle)

                inst_mask = inst_mask[ymin:ymax, xmin:xmax]
                new_inst_mask = __transform_img(inst_mask, trans_param, (height, width))

                rle = cocomask.encode(new_inst_mask)
                new_segm = binary_mask_to_rle(inst_mask)
                bbox = cocomask.toBbox(rle).tolist()
            else:
                # coco-style RLE -- error
                raise ValueError('You should not input json containing RLE annotations!')

            ann['segmentation'] = new_segm
            ann['bbox'] = bbox
            ret_anns.append(ann)

    return ret_anns
