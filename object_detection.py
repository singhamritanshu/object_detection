import cv2 as cv
thresh = 0.45 # Threshold to detect object

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)
# Getting all the class names from the coco.names
classNames = []
classFiles = 'coco.names'
with open(classFiles,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

net = cv.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thresh) # detect will give us the classId, confidence score and the bounding box co-ordinates
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv.rectangle(img,box, color=(0,255,0), thickness=2)
            cv.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv.imshow("Output",img)
    cv.waitKey(1)