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
import time


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
    imgResized = cv2.resize(img, (360, 360))
    imgResizedName = "test_%d.jpg" % i
    cv2.imwrite(imgResizedName, imgResized)
    LpImg, LpType, cor = get_plate(imgResizedName)
    print("Detected %i  plate(s) " % len(LpImg))
    if len(LpImg):  # check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=255.0)
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Applied inverse thresh_binary, hiểu nôm na là chuyển sang ảnh nhị phân
        binary = cv2.threshold(blur, 180, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # cv2.imshow("Anh bien so sau chỉnh sua", binary)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        #cv2.imshow("thre_mor", thre_mor)

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

    for c in sort_contours(cont):  # now become cont
        (x, y, w, h) = cv2.boundingRect(c)
        area = w*h
        #print(area)
        ratio = h / w
        if (1 <= ratio <= 3.5)&(area >=1500):  # Only select contour with defined ratio
            if h / plate_image.shape[0] >= 0.35:  # Select contour which has the height larger than 50% of the plate
                # Draw bounding box around digit number
                #cv2.rectangle(tested_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Separate number and give prediction
                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)
    # cv2.imshow("Ket qua", tested_plate)
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

if __name__ == "__main__":
    # test = './img1.jpg'
    # test1 = './img2.jpg'
    startTime = time.time()
    plate = plateDetect('img1.jpg', 0)
    print(plate)
    print("--- Detected in %s seconds ---" % str(time.time()-startTime))
    firstTime = time.time()
    plate1 = plateDetect('img1.jpg', 1)
    print(plate1)
    print("--- Detected in %s seconds ---" % str(time.time()-firstTime))
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