# https://imageai.readthedocs.io/en/latest/video/

from imageai.Detection import VideoObjectDetection
import cv2

# camera = cv2.VideoCapture(0)

pretrained_model = "retinanet.pth"
input = "input_video.gif"
output = "output_video"

detector = VideoObjectDetection()

detector.setModelTypeAsRetinaNet()  

detector.setModelPath(pretrained_model)  
 
detector.loadModel()

# custom = detector.CustomObjects(person=True, dog=True)
video_path = detector.detectObjectsFromVideo(input_file_path=input, output_file_path=output, frames_per_second=20, log_progress=True, minimum_percentage_probability=30)
# video_path = detector.detectObjectsFromVideo(camera_input=camera, output_file_path=output, frames_per_second=20, log_progress=True, minimum_percentage_probability=30)

print(video_path)