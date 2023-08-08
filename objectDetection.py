from imageai.Detection import ObjectDetection
from collections import OrderedDict  

def objectDetection():
    #Object DEtection From Image 
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("models/yolo.h5")
    detector.loadModel()
    detections = detector.detectObjectsFromImage("images/uploadImage.jpeg","images/ResultObject.jpeg")
    myData = []
    for eachObject in detections:
        #print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
        myData.append(eachObject["name"])
    entityList = list(OrderedDict.fromkeys(myData))
    quantityList = {item: myData.count(item) for item in myData}
    print(entityList,quantityList)
    return entityList,quantityList