import numpy as np
import cv2


def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle

        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scale_fill:
        dw, dh = 0, 0
        new_unpad = new_shape[::-1]  # [h, w] -> [w, h]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # wh ratios

    dw /= 2
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)








