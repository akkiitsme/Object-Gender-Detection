from flask import Flask, jsonify, request 
import numpy as np
import io,cv2,base64
from PIL import Image
from objectDetection import objectDetection
from genderDetection import genderDetect


#API Generation 
app = Flask(__name__)

@app.route('/api/imagedetection')
def objectResult():
    #request.form['name']
    localImage = request.form['image']

    #converting base 64 to image
    imgdata = base64.b64decode(str(localImage))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    cv2.imwrite("images/uploadImage.jpeg",opencv_img)


    #Object Detection from image
    entityList,quantityList=objectDetection()
    #Gender and Age Detection from image
    age,gender=genderDetect()

    if objectResult is not None and age is not None and gender is not None :
        result={
        "message" : "success",
        "entities": entityList,
        "quantity": quantityList,
        "gender": gender,
        "age":age,
        "status": True
        }
    elif objectResult is None and age and gender is not None:
        result={
        "message" : "success",
        "entities": "",
        "quantity": "",
        "gender": gender,
        "age": age,
        "status": True 
        }
    elif objectResult is not None and age is None and gender is None :
        result={
        "message" : "success",
        "entities": entityList,
        "quantity": quantityList,
        "gender": "",
        "age": "",
        "status": True  
        }
    elif objectResult is None and age is None and gender is None : 
        result={
        "message" : "success",
        "entities": "Empty",
        "quantity": "Empty",
        "gender": "Empty",
        "age": "Empty",
        "status": False  
        } 
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)