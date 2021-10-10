from typing import List
import cv2
import timeit
import numpy as np
from sklearn.linear_model import SGDClassifier

def local_vid(name):
    return "data_raw/video/" + name + ".mp4";
def local_pic(name,folder=""):
    if folder == "":
        return "data_raw/picture/" + name +".jpg";
    else:
        return "data_raw/picture/" + folder +"/"+ name +".jpg";

def showim(img):
    cv2.imshow("show",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def picture_from_video(filename,perfream,amoung=1):
    start = timeit.default_timer()
    end_NO = 0
    for i in range(amoung):
        cap = cv2.VideoCapture(local_vid(filename+str(i+1)))
        for fream_index in range(50000000000):
            try:
                ret,pixel = cap.read()
                if fream_index%perfream == 0:
                    cv2.imwrite(local_pic("pic_"+str(fream_index//perfream+end_NO),folder="testset"),cv2.cvtColor(pixel, cv2.COLOR_BGR2GRAY)[180:][:])
            except:
                end_NO = (fream_index//perfream) + end_NO
                break
        print("video " +str(i+1)+ " picture complete")
    print("Finish picture form Video")
    stop = timeit.default_timer()
    print('Time: ', stop - start)

def data_from_video(filename,perfream,amoung=1):
    start = timeit.default_timer()
    output_data = []
    end_NO = 0
    for i in range(amoung):
        cap = cv2.VideoCapture(local_vid(filename+str(i+1)))
        for fream_index in range(50000000000):
            try:
                ret,pixel = cap.read()
                if fream_index%perfream == 0:
                    output_data.append(cv2.cvtColor(pixel, cv2.COLOR_BGR2GRAY)[180:][:].flatten()) 
            except:
                end_NO = (fream_index//perfream) + end_NO
                break
        print("video " +str(i+1)+ " clear")
    print("--- Finish Data form Video ---")
    stop = timeit.default_timer()
    print('--- Time: ', stop - start)
    return np.array(output_data)

def data_form_picture(name):
    img = cv2.imread(local_pic(name,folder="testset"),0)
    return img

