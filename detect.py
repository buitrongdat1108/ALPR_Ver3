import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import socket
import cv2
import numpy as np
from local_utils import predict_from_model, get_plate
from keras.models import model_from_json
#from keras.preprocessing.image import load_img, img_to_array
#from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from os.path import splitext
import time


class Label:
    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top left(x: %f, y: %f), bottom right(x: %f, y: %f)' % (
            self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl

    def cc(self): return self.__tl + self.wh() / 2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    def bl(self): return np.array([self.__tl[0], self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob


class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, axis=1)
        br = np.amax(pts, axis=1)
        Label.__init__(self, cl, tl, br, prob)


# Hàm normalize ảnh
def im2single(Image):
    return Image.astype('float32') / 255


def getWH(shape):
    return np.array(shape[1::-1]).astype(float)


def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1 - tl1, br2 - tl2
    assert ((wh1 >= 0).all() and (wh2 >= 0).all())

    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area


def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())


def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)
    return SelectedLabels


def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T

        A[i * 2, 3:6] = -xil[2] * xi
        A[i * 2, 6:] = xil[1] * xi
        A[i * 2 + 1, :3] = xil[2] * xi
        A[i * 2 + 1, 6:] = -xil[0] * xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H


def getRectPts(tlx, tly, brx, bry):
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1, 1, 1, 1]], dtype=float)


def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop


# Reconstruction function from predict value into plate crpoped from image
# Hàm tái tạo từ predict value thành biển số, cắt từ ảnh chính ra biển số, nhãn,...
def reconstruct(I, Iresized, Yr, lp_threshold):
    # 4 max-pooling layers, stride = 2
    net_stride = 2 ** 4
    side = ((208 + 40) / 2) / net_stride

    # one line and two lines license plate size
    one_line = (470, 110)
    two_lines = (280, 200)

    Probs = Yr[..., 0]
    Affines = Yr[..., 2:]

    xx, yy = np.where(Probs > lp_threshold)
    # CNN input image size
    WH = getWH(Iresized.shape)
    # output feature map size
    MN = WH / net_stride

    vxx = vyy = 0.5  # alpha
    base = lambda vx, vy: np.matrix([[-vx, -vy, 1], [vx, -vy, 1], [vx, vy, 1], [-vx, vy, 1]]).T
    labels = []
    labels_frontal = []

    for i in range(len(xx)):
        x, y = xx[i], yy[i]
        affine = Affines[x, y]
        prob = Probs[x, y]

        mn = np.array([float(y) + 0.5, float(x) + 0.5])

        # affine transformation matrix
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0)
        A[1, 1] = max(A[1, 1], 0)
        # identity transformation
        B = np.zeros((2, 3))
        B[0, 0] = max(A[0, 0], 0)
        B[1, 1] = max(A[1, 1], 0)

        pts = np.array(A * base(vxx, vyy))
        pts_frontal = np.array(B * base(vxx, vyy))

        pts_prop = normal(pts, side, mn, MN)
        frontal = normal(pts_frontal, side, mn, MN)

        labels.append(DLabel(0, pts_prop, prob))
        labels_frontal.append(DLabel(0, frontal, prob))

    final_labels = nms(labels, 0.1)
    final_labels_frontal = nms(labels_frontal, 0.1)

    # print(final_labels_frontal)
    assert final_labels_frontal, "No License plate is founded!"

    # LP size and type
    out_size, lp_type = (two_lines, 2) if (
            (final_labels_frontal[0].wh()[0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)

    TLp = []
    Cor = []
    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for _, label in enumerate(final_labels):
            t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
            ptsh = np.concatenate((label.pts * getWH(I.shape).reshape((2, 1)), np.ones((1, 4))))
            H = find_T_matrix(ptsh, t_ptsh)
            Ilp = cv2.warpPerspective(I, H, out_size, borderValue=0)
            TLp.append(Ilp)
            Cor.append(ptsh)
    return final_labels, TLp, lp_type, Cor


def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("[INFO] Wpod-Net model loaded successfully...")
        return model
    except Exception as e:
        print(e)


wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)


def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image,) * 3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])
    return prediction


def detect_lp(model, I, max_dim, lp_threshold):
    # Tính factor resize ảnh
    min_dim_img = min(I.shape[:2])
    factor = float(max_dim) / min_dim_img

    # Tính W và H mới sau khi resize
    w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()
    # Tiến hành resize ảnh
    Iresized = cv2.resize(I, (w, h))
    T = Iresized.copy()
    # Chuyển thành tensor
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    # Tiến hành detect biển số bằng wpod-net pretrain
    Yr = model.predict(T)
    Yr = np.squeeze(Yr)  # remove các chiều = 1 của Yr
    # print(Yr.shape)
    L, TLp, lp_type, cor = reconstruct(I, Iresized, Yr, lp_threshold)
    return L, TLp, lp_type, cor


def get_plate(image_path, Dmax=700, Dmin=500):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])  # vehicle.shape[:2] = (720,1280), ratio = 1.77777778
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, LpType, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    # print(LpType)
    return LpImg, LpType, cor

#PART 1: Load model architecture, weight and labels

json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] MobileNet model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")

def paddingImg(inputImg, resize=False):
    padding = np.zeros([inputImg.shape[1], inputImg.shape[1], 3])
    a = int((inputImg.shape[1] - inputImg.shape[0]) / 2)
    padding[a:(inputImg.shape[0] + a), 0:inputImg.shape[1], :] = inputImg
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)
    inputImg = inputImg / 255
    if resize:
        inputImg = cv2.resize(inputImg, (720, 720))
    return inputImg

#PART 2: Detect plate image and segment character
def plateDetect(testImagePath, i = None):
    '''
     #crop
    img = cv2.imread(testImagePath)
    imgCrop = img[540:900, 395:755, :]
    imgCropName = 'test_%d.jpg' % i
    cv2.imwrite(imgCropName, imgCrop)
    '''
    '''
    
    '''
    # resize
    img = cv2.imread(testImagePath)
    imgResized = cv2.resize(img, (1280, 720))
    imgResizedName = "test_%d.jpg" % i
    cv2.imwrite(imgResizedName, imgResized)
    LpImg, LpType, cor = get_plate(imgResizedName)
    pts = []
    x_coordinates = cor[0][0]
    y_coordinates = cor[0][1]
    # store the top-left,top-right,bottom-left,bottom-right(tl-tr-br-bl)
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]), int(y_coordinates[i])])
    pts = np.array(pts, np.int32) #lấy được tọa độ 4 góc của BSX
    print("Detected %i  plate(s) " % len(LpImg))
    if len(LpImg):  # check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=255.0)
        print(type(plate_image))
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Applied inverse thresh_binary, hiểu nôm na là chuyển sang ảnh nhị phân
        binary = cv2.threshold(blur, 180, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # cv2.imshow("Anh bien so sau chỉnh sua", binary)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cv2.imshow("thre_mor", thre_mor)

    # Create sort_contours() function to grab the contour of each digit from left to right
    def sort_contours(cnts, reverse=False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        # print(boundingBoxes)
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return cnts
    cont, _ = cv2.findContours(thre_mor, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list which will be used to append character image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60
    tested_plate = plate_image.copy()
    for c in sort_contours(cont):  # now become cont
        (x, y, w, h) = cv2.boundingRect(c)
        area = w*h
        #print(area)
        ratio = h / w
        if (1 <= ratio <= 3.5):  # Only select contour with defined ratio
            if h / plate_image.shape[0] >= 0.35:  # Select contour which has the height larger than 50% of the plate
                #Draw bounding box around digit number

                cv2.rectangle(tested_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Separate number and give prediction
                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)
    cv2.imshow("Ket qua", tested_plate)
    print("Detected {} letters...".format(len(crop_characters)))

    # pre-processing input images and predict with model

    final_string = ''
    for i, character in enumerate(crop_characters):
        title = np.array2string(predict_from_model(character, model, labels))
        final_string += title.strip("'[]")
        final_string1 = final_string
        final_string1_List = list(final_string1)
        #final_string1_List.reverse()
        final_string1 = ''.join(final_string1_List)
    return final_string1

def testImg(imgPath):
    img = cv2.imread(imgPath)
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=90, scale=1)
    rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))
    cv2.imwrite("img4_rotated.jpg", rotated_image)
    return plateDetect("img4_rotated.jpg", 1)
if __name__ == "__main__":
    # test = './img1.jpg'
    # test1 = './img2.jpg'
    startTime = time.time()
    plate = plateDetect('/home/vdtcdatbt/ALPR/ALPR_Ver5/tested_image/img14.jpg', 0)
    print(plate)
    print("--- Detected in %s seconds ---" % str(time.time()-startTime))
    # firstTime = time.time()
    # plate1 = plateDetect('/home/vdtcdatbt/ALPR/ALPR_Ver5/tested_image/img14.jpg', 1)
    # print(plate1)
    # print("--- Detected in %s seconds ---" % str(time.time()-firstTime))
    cv2.waitKey()

# #PART 3: Create socket and send detected information

# serverSocket = socket.socket()
# serverSocket.bind(('', 8888))
# serverSocket.listen(10)


# while True:
#     (client, addr) = serverSocket.accept()
#     receivedPlateDir = client.recv(1024)
#     startTime = time.time()
#     plateNum = plateDetect(receivedPlateDir.decode())
#     client.send(plateNum.encode())
#     endTime = time.time() - startTime
#     timeReq = client.recv(1024)
#     client.send(str(endTime).encode())