# coding:utf-8

import math
import random
import numpy as np

'''
action:[whatToDo, zoom_factor, delta_x, delta_y, beta]
'''


class Trans:

    def __init__(self, image, action):
        self.src = image  # 原始图像
        self.rows = self.src.shape[0]  # 原始图像的行
        self.cols = self.src.shape[1]  # 原始图像的列
        self.center = [0, 0]  # 中心，默认是[0,0]
        self.dst = np.zeros((self.rows, self.cols, 4), dtype=np.uint8)  # 最终图像

        whatToDo = action["whatToDo"]
        zoom_factor = action["zoom_factor"]
        delta_x = action["delta_x"]
        delta_y = action["delta_y"]
        beta = action["beta"]

        if whatToDo == "skip":
            self.dst = self.src
            self.transform = None
            return
        if whatToDo == "horizontal":
            self.Horizontal()
        if whatToDo == "vertical":
            self.Vertical()
        if whatToDo == "normal":
            self.center = [self.rows//2, self.cols//2]
            self.transform = zoom_factor, beta, delta_x, delta_y
            '''
            np.array([ \
                [zoom_factor*math.cos(beta), -math.sin(beta), delta_x], \
                [math.sin(beta), zoom_factor*math.cos(beta), delta_y], \
                [0, 0, 1]])
            '''

    def Move(self, delta_x, delta_y):  # 平移,center=[rows//2,cols//2]
        # delta_x>0左移，delta_x<0右移
        # delta_y>0上移，delta_y<0下移
        self.transform = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])

    def Zoom(self, factor):  # 缩放,center=[rows//2,cols//2]
        # factor>1表示缩小；factor<1表示放大
        self.transform = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])

    def Horizontal(self):  # 水平镜像,center=[0,0]
        self.transform = np.array([[1, 0, 0], [0, -1, self.cols - 1], [0, 0, 1]])

    def Vertical(self):  # 垂直镜像,center=[0,0]
        self.transform = np.array([[-1, 0, self.rows - 1], [0, 1, 0], [0, 0, 1]])

    def Rotate(self, beta):  # 旋转
        # beta>0表示逆时针旋转；beta<0表示顺时针旋转
        self.transform = np.array([[math.cos(beta), -math.sin(beta), 0],
                                   [math.sin(beta), math.cos(beta), 0],
                                   [0, 0, 1]])

    def Process(self):
        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(4):
                    src_pos = np.array([i - self.center[0], j - self.center[1], 1])
                    [x, y, z] = np.dot(self.transform, src_pos)
                    x = int(x) + self.center[0]
                    y = int(y) + self.center[1]

                    if x >= self.rows or y >= self.cols or x < 0 or y < 0:
                        self.dst[i][j] = 255
                    else:
                        self.dst[i][j] = self.src[x][y]


def get_transform(src, restricts):
    if 'noflip' in restricts and restricts['noflip'] == 1:
        action_candidate = ['normal', 'skip']
    else:
        action_candidate = ['normal', 'horizontal', 'skip', 'skip']

    action_what = random.choice(action_candidate)

    if action_what == 'skip':
        t = __identity_transform()
    elif action_what == 'horizontal':
        src = src[:, ::-1, :]  # horizontal flip
        t = __random_transform(restricts)
        t['flip'] = 'horizontal'
    elif action_what == 'vertical':
        src = src[::-1, :, :]  # vertical flip
        t = __random_transform(restricts)
        t['flip'] = 'vertical'
    elif action_what == 'normal':
        t = __random_transform(restricts)
    else:
        raise ValueError('Unknown action {}'.format(action_what))

    return src, t


def __random_transform(restricts):
    t = dict()
    t['s'] = random.uniform(0.5, 2)
    max_x = restricts['bbox_w'] // 5
    t['tx'] = random.randint(-max_x, max_x)
    max_y = restricts['bbox_h'] // 5
    t['ty'] = random.randint(-max_y, max_y)
    t['theta'] = math.radians(random.randint(-30, 30))

    if 'restrict_x' in restricts and restricts['restrict_x'] == 1:
        t['s'] = 1
        t['tx'] = 0
        t['theta'] = 0
    if 'restrict_y' in restricts and restricts['restrict_y'] == 1:
        t['s'] = 1
        t['ty'] = 0
        t['theta'] = 0

    return t


def __identity_transform():
    t = dict()
    t['s'] = 1
    t['tx'] = 0
    t['ty'] = 0
    t['theta'] = 0
    return t


def get_restriction(bndbox, width, height):
    """
    Restrict transform parameters.
    :param bndbox: bounding box of original object in [xmin, ymin, xmax, ymax]
    :param width: image width
    :param height: image height
    :return: a dictionary containing restrictions
    """
    xmin, ymin, xmax, ymax = bndbox
    restricts = dict()
    restricts['bbox_w'] = xmax - xmin
    restricts['bbox_h'] = ymax - ymin
    if xmin < 10 or xmax > width - 10:
        restricts['restrict_x'] = 1
        restricts['noflip'] = 1
    if ymin < 10 or ymax > height-10:
        restricts['restrict_y'] = 1
        restricts['noflip'] = 1
    return restricts
