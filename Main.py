import cv2
import Basefunction
import timeit
import numpy as np
from sklearn.linear_model import SGDClassifier
import joblib

def showim(img):
    cv2.imshow("show",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def data_from_video(filename,perfream,amoung=1):
    start = timeit.default_timer()
    output_data = []
    end_NO = 0
    for i in range(amoung):
        cap = cv2.VideoCapture(Basefunction.local_vid(filename+str(i+1)))
        for fream_index in range(50000000000):
            try:
                ret,pixel = cap.read()
                if fream_index%perfream == 0:
                    output_data.append(pixel.flatten()) 
            except:
                end_NO = (fream_index//perfream) + end_NO
                break
        print("video " +str(i+1)+ " clear")
    print("Finish Data form Video")
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return np.array(output_data)
    

def label(true_index,size):
    label_list = []
    for i in range(size):
        if i in true_index:
            label_list.append(True)
        else:
            label_list.append(False)
    print("Finish label")
    return label_list;

def gain_model():
    number_data = 1338
    true_no = [51,52,156,157,373,374,548,549,645,646,936,937,1028,1029,1275,1276]

    answer = label(true_no,number_data)
    data_list = data_from_video(number_data)

    sgd = SGDClassifier()
    sgd.fit(data_list,answer)
    joblib.dump(sgd,"model1.joblib")

data = data_from_video("data",72,amoung=4)
print(data.shape)