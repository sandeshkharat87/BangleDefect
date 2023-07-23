## Visulization Tools
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("/content/content/runs/detect/train2/weights/best.pt")

def makePred(img):
    x = cv2.resize(cv2.imread(img),(640,640))
    # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    rr = model.predict(x)
    rr = rr[0].boxes.xyxy.tolist()
    if rr:
        for cords in rr:
            x1,y1,x2,y2 = np.array(cords).astype(np.int32)
            frame = cv2.rectangle(x, (x1,y1),(x2,y2),(255,12,12),4  )
            frame = cv2.putText(frame,text="Defect", org=(x1,y1-10), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale=2, color=(0,123,123),thickness=3  )

        return frame
    return False

ResultImage = makePred("/content/img7663.jpg")

cv2.imwrite("ResultImage.jpg",ResultImage)
