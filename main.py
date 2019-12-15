import sys
from abc import ABCMeta, abstractmethod

import numpy as np
import cv2
import cvui

class CvuiComponent(metaclass=ABCMeta):
    def __init__(self):
        self.frame = None

    @abstractmethod
    def process(self):
        pass

class Trackbar(CvuiComponent):
    def __init__(self, x, y, w, defval, minval, maxval, segments, format):
        self.value = [defval]
        self.x = x
        self.y = y
        self.w = w
        self.minval = minval
        self.maxval = maxval
        self.segments = segments
        self.format = format

        self.on_value_changed = None

    def val(self):
        return self.value[0]
    
    def process(self):
        changed = cvui.trackbar(self.frame, self.x, self.y, self.w, self.value,
                        self.minval, self.maxval, self.segments, self.format)
        if changed and self.on_value_changed:
            self.on_value_changed(self.val())
        

class Image(CvuiComponent):
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image

    def process(self):
        cvui.image(self.frame, self.x, self.y, self.image)

class Text(CvuiComponent):
    def __init__(self, x, y, text):
        self.x = x
        self.y = y
        self.text = text

    def process(self):
        cvui.text(self.frame, self.x, self.y, self.text)

class CvuiMainFrame:

    BACKGROUND_COLOR = (49, 52, 49)

    def __init__(self, window_name, width, height, delay=10):
        self.__frame = np.zeros((height, width, 3), np.uint8)
        self.__window_name = window_name
        self.__components = []
        self.__delay = delay
    
    def append(self, component):
        component.frame = self.__frame
        self.__components.append(component)
    
    def start(self):
        cvui.init(self.__window_name)

        while True:
            self.__frame[:] = CvuiMainFrame.BACKGROUND_COLOR

            for c in self.__components:
                c.process()

            cvui.update()
            cv2.imshow(self.__window_name, self.__frame)

            if cv2.waitKey(self.__delay) == 27: # = ESC
                break

def create_gabor_image(src_img, param, size):
    gabor = calc_gabor(param)
    dst_img = cv2.filter2D(src_img, -1, gabor)
    gabor_img = create_gabor_bgr_image(gabor, size)
    return (dst_img, gabor_img, gabor)

def calc_gabor(param):
    k = int(param.ksize)
    ksize = (k, k)
    sigma = param.sigma
    theta = (np.pi / 180) * param.angle
    lambd = param.lambd
    gamma = param.gamma
    return cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma)

def create_gabor_bgr_image(gabor, size):
    gabor_image = gabor.copy()
    cv2.normalize(gabor, gabor_image, 0, 255, cv2.NORM_MINMAX)
    gabor_image = gabor_image.astype(np.uint8)
    gabor_image = cv2.cvtColor(gabor_image, cv2.COLOR_GRAY2BGR)
    gabor_image = cv2.resize(gabor_image, size)
    return gabor_image

class GaborParams:
    def __init__(self, ksize, sigma, angle, lambd, gamma):
        self.ksize = ksize
        self.sigma = sigma
        self.angle = angle
        self.lambd = lambd
        self.gamma = gamma

GABOR_IMG_SIZE = (200, 200)
WINDOW_NAME = 'Gabor'

def main(filepath):
    mainframe = CvuiMainFrame(WINDOW_NAME, 900, 800)

    src_img = cv2.imread(filepath)
    target_img = cv2.resize(src_img, None, fx=0.3, fy=0.3)

    gabor_param = GaborParams(10, 2.0, 0, 10, 0.5)
    target_gabor_img, gabor_img, _ = create_gabor_image(target_img, gabor_param, GABOR_IMG_SIZE)

    gabor_img_area = Image(30, 500, gabor_img)
    target_img_area = Image(300, 10, target_img)
    target_gabor_img_area = Image(300, 360, target_gabor_img)

    kernel_trackbar = Trackbar(20,  40, 250,  10,   1,  100, 1,   '%d')
    sigma_trackbar  = Trackbar(20, 120, 250, 2.0, 0.1, 10.0, 1, '%.1f')
    angle_trackbar  = Trackbar(20, 200, 250,   0,   0,  180, 1,   '%d')
    lambd_trackbar  = Trackbar(20, 280, 250,  10,   0,  100, 1,   '%d')
    gamma_trackbar  = Trackbar(20, 360, 250, 0.1,   0, 10.0, 1, '%.1f')

    def gabor_value_chanded(v, param_name):
        nonlocal gabor_param
        gabor_param.__dict__[param_name] = v
        target_gabor_img_area.image, gabor_img_area.image, _ = create_gabor_image(target_img, gabor_param, GABOR_IMG_SIZE)
    
    kernel_trackbar.on_value_changed = lambda v: gabor_value_chanded(v, 'ksize')
    sigma_trackbar.on_value_changed = lambda v: gabor_value_chanded(v, 'sigma')
    angle_trackbar.on_value_changed = lambda v: gabor_value_chanded(v, 'angle')
    lambd_trackbar.on_value_changed = lambda v: gabor_value_chanded(v, 'lambd')
    gamma_trackbar.on_value_changed = lambda v: gabor_value_chanded(v, 'gamma')

    mainframe.append(target_img_area)
    mainframe.append(target_gabor_img_area)
    mainframe.append(gabor_img_area)
    mainframe.append(Text(10, 20, 'Kernel'))
    mainframe.append(kernel_trackbar)
    mainframe.append(Text(10, 100, 'Sigma'))
    mainframe.append(sigma_trackbar)
    mainframe.append(Text(10, 180, 'Angle'))
    mainframe.append(angle_trackbar)
    mainframe.append(Text(10, 260, 'Lambd'))
    mainframe.append(lambd_trackbar)
    mainframe.append(Text(10, 340, 'Gamma'))
    mainframe.append(gamma_trackbar)

    mainframe.start()

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        # TODO: error message
        sys.exit(1)
    main(args[1])

