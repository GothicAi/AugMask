import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visual_anns(img, anns, coco):
    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    coco.showAnns(anns)
    ax = plt.gca()
    for ann in anns:
        bbox = ann['bbox']
        ax.add_patch(
            matplotlib.patches.Rectangle(
                bbox[0:2], bbox[2], bbox[3],
                fill=False, edgecolor='g', linewidth=1
            )
        )
        area = ann['area']
        print(area)
    plt.savefig('ann.jpg')
    plt.close()
