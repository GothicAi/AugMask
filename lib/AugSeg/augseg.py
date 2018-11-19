import numpy as np
from get_instance_group import extract
from affine_transform import transform_image, transform_annotation


def get_new_data(ori_anns: list, ori_img: np.ndarray):
    """
    Get a new image with new annotations from original image and annotations
    :param ori_anns: list of coco-style annotation dicts
    :param ori_img: numpy array
    :return: new_ann, new_img
    """
    background, instances_list, transforms_list, groupbnds_list, groupidx_list = extract(ori_anns, ori_img)
    new_img = transform_image(background, instances_list, transforms_list)
    new_ann = transform_annotation(ori_anns, transforms_list, groupbnds_list, groupidx_list,
                                   background.shape[1], background.shape[0])
    return new_ann, new_img


if __name__ == '__main__':
    from pycocotools.coco import COCO
    import os
    from scipy.misc import imread, imsave
    from visual_anns import visual_anns

    imgdir = '/data/jianhua/cocoBackup/images/val2017/'
    coco = COCO('/data/jianhua/cocoBackup/annotations_coco/instances_val2017.json')
    ImgIds = coco.getImgIds()

    ImgIds = ImgIds[:10]

    for img_id in ImgIds:
        IMG = coco.loadImgs(img_id)
        filename = IMG[0]['file_name'].strip('.jpg')
        img_path = os.path.join(imgdir, filename + '.jpg')
        assert os.path.exists(img_path), '{} does not exist.'.format(img_path)
        image = imread(img_path)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        new_anns, new_img = get_new_data(anns, image)

        oriimg = imread(imgdir + coco.loadImgs(img_id)[0]['file_name'])
        imsave('oriimg.jpg', oriimg)
        imsave('newimg.jpg', new_img)
        visual_anns(new_img, new_anns, coco)
        pass