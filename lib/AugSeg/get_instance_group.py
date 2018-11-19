import matplotlib
matplotlib.use('AGG')
    
import numpy as np
import cv2
import pycocotools.mask as cocomask
import opencv_mat as gm
from single_image_process import get_transform, get_restriction


def __cocoseg_to_binary(seg, height, width):
    """
    COCO style segmentation to binary mask
    :param seg: coco-style segmentation
    :param height: image height
    :param width: image width
    :return: binary mask
    """
    if type(seg) == list:
        rle = cocomask.frPyObjects(seg, height, width)
        rle = cocomask.merge(rle)
        mask = cocomask.decode([rle])
    elif type(seg['counts']) == list:
        rle = cocomask.frPyObjects(seg, height, width)
        rle = cocomask.merge(rle)
        mask = cocomask.decode([rle])
    else:
        rle = cocomask.merge(seg)
        mask = cocomask.decode([rle])
    assert mask.shape[2] == 1
    return mask[:, :, 0]


def __get_coco_masks(anns: list, height: int, width: int):
    """
    Get coco masks from annotations.
    :param anns: list of coco-style annotation
    :param height: image height
    :param width: image width
    :return: masks, hxw numpy array
             classes, nx1
    """
    if len(anns) == 0:
        return None, None
    
    classes = []
    mask = np.zeros((height, width), dtype=np.int32)
    
    for inst_idx, ann in enumerate(anns):
        m = __cocoseg_to_binary(ann['segmentation'], height, width)  # zero one mask
        cat_id = ann['category_id']
        classes.append(cat_id)
        
        m = m.astype(np.int32) * (inst_idx + 1)
        mask[m > 0] = m[m > 0]

    classes = np.asarray(classes)
    return mask, classes


def get_mask(mat, k):
    retMat = (mat - 1 == k) * 255
    return retMat.astype(np.uint8)
    
    
# input 2 bboxes, return whether the two bboxes overlap
def overlap(bbox1,bbox2):
    [x1,y1,w1,h1] = bbox1
    [x2,y2,w2,h2] = bbox2
    return (x1 < x2 + w2) and (y1 < y2 + h2) and (x2 < x1 + w1) and (y2 < y1 + h1)
    

def DFS(bboxs,groups,groupi,g,i):
    numinst = len(bboxs)
    for j in range(numinst):
        if groups[j] == 0 and j != i:
            if overlap(bboxs[i],bboxs[j]):
                groups[j] = g[0]
                groupi.append(j)
                DFS(bboxs,groups,groupi,g,j)


# k is the thickness of the alphamap
def gettrimap(mask, k):
    w, h = mask.shape
    np.set_printoptions(threshold=np.nan)
    trimap = np.zeros([w, h], dtype=np.uint8)
    for n in range(w):
        for m in range(h):
            xmin = max([n - k, 0])
            ymin = max([m - k, 0])
            xmax = min([n + k, w])
            ymax = min([m + k, h])
            neighbor = mask[xmin:xmax,ymin:ymax]
            # print(neighbor)
            if neighbor.max() != neighbor.min() and mask[n,m] == 0:
                trimap[n,m] = 127
            else:
                trimap[n,m] = mask[n,m]
    assert trimap.max() == 255 and trimap.min() == 0, 'trimap error'
    return trimap


def get_masks(mat, klist):
    xlen, ylen = mat.shape
    retMat = np.zeros((xlen, ylen), dtype=np.uint8)
    for k in klist:
        retMat += (mat - 1 == k).astype(np.uint8) * 255
    return retMat
    

def extract(anns, img):

    width = img.shape[1]
    height = img.shape[0]

    mask, labels = __get_coco_masks(anns, height, width)
    if mask is None:
        return

    # inpainting
    background = cv2.inpaint(img, np.uint8(mask), 5, cv2.INPAINT_NS)

    numinst = mask.max()
    bboxs = []

    for ann in anns:
        x, y, w, h = ann['bbox']
        ymin = y
        xmin = x
        ymax = y + h
        xmax = x + w
        bboxs.append(ann['bbox'])

    groups = [0] * numinst
    group = []
    g = [1]
    for i in range(numinst):
        if groups[i] == 0:
            group.append([i])
            groups[i] = g[0]
            DFS(bboxs, groups, group[len(group) - 1], g, i)
            g[0] += 1
    realbboxs = []

    instances_list = []
    transforms_list = []
    groupbnd_list = []
    for i in range(len(group)):
        x, y, w, h = bboxs[group[i][0]]
        realbboxs.append([x, y, x + w, y + h])
        for j in range(len(group[i])):
            x, y, w, h = bboxs[group[i][j]]
            realbboxs[i][0] = min(realbboxs[i][0], x)
            realbboxs[i][1] = min(realbboxs[i][1], y)
            realbboxs[i][2] = max(realbboxs[i][2], x + w)
            realbboxs[i][3] = max(realbboxs[i][3], y + h)
            xmin, ymin, xmax, ymax = realbboxs[i]

        maskgroupi = get_masks(mask, group[i])
        try:
            trimapi = gettrimap(maskgroupi, 5)
        except AssertionError:
            print('Trimap error, skipping...')
            return
        alphamapi = gm.global_matting(img, trimapi)
        alphamapi = gm.guided_filter(img, trimapi, alphamapi, 10, 1e-5)
        ymin, ymax, xmin, xmax = [int(round(x)) for x in (ymin, ymax, xmin, xmax)]
        r, g, b = cv2.split(img[ymin:ymax, xmin:xmax])

        assert alphamapi[ymin:ymax, xmin:xmax].shape == g.shape
        alphamapiSmall = alphamapi[ymin:ymax, xmin:xmax]
        resulti = cv2.merge((r, g, b, alphamapiSmall))

        restricts = get_restriction([xmin, ymin, xmax, ymax], width, height)
        resulti, transformi = get_transform(resulti, restricts)  # resulti may be flipped
        transformi['tx'] += (xmin + xmax) / 2
        transformi['ty'] += (ymin + ymax) / 2
        instances_list.append(resulti)
        transforms_list.append(transformi)
        groupbnd_list.append([xmin, ymin, xmax, ymax])

    return background, instances_list, transforms_list, groupbnd_list, group
