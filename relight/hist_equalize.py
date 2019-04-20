import cv2

def rgb_equalized(image):
    eq_channels = [cv2.equalizeHist(ch) for ch in cv2.split(image)]
    eq_image = cv2.merge(eq_channels)
    return eq_image

def hsv_equalized(image):
    H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image


# CLAHE 
def img_clahe(image):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    eq_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return eq_image


