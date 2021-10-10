from typing import List
import cv2
from basefunction import *
from sklearn.linear_model import SGDClassifier
import joblib

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
    number_data = 1380
    true_no = [51,52,156,157,373,374,548,549,639,640,886,887,1040,1041,1331,1332]

    answer = label(true_no,number_data)
    print("answer size = " + str(len(answer)))
    data_list = data_from_video("data",72,amoung=4)
    print("datalist size = " + str(data_list.shape))
    print("start train model")
    sgd = SGDClassifier()
    sgd.fit(data_list,answer)
    print("end train model")
    joblib.dump(sgd,"model1.joblib")
    print("finish")