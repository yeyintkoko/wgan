# https://imageai.readthedocs.io/en/latest/

from imageai.Detection import ObjectDetection

pretrained_model = "retinanet.pth"
input = "input_image.jpg"
output = "output_image.jpg"

detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()  

detector.setModelPath(pretrained_model)  
 
detector.loadModel()

custom = detector.CustomObjects(person=True, dog=True)
detections = detector.detectObjectsFromImage(custom_objects=custom, input_image=input, output_image_path=output, minimum_percentage_probability=30)
# detections = detector.detectObjectsFromImage(input_image=input, output_image_path=output, minimum_percentage_probability=30)

# iterating through the items found in the image
for eachItem in detections:  
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])  