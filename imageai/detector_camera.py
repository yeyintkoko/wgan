from imageai.Detection import VideoObjectDetection
import cv2
from matplotlib import pyplot as plt

# Initialize the camera
camera = cv2.VideoCapture(0)

# Set the pretrained model path
pretrained_model = "retinanet.pth"
output = "output_video"

# Initialize the detector object
detector = VideoObjectDetection()

# Set the model type and load the model
detector.setModelTypeAsRetinaNet()
detector.setModelPath(pretrained_model)
detector.loadModel()

# Define a function for frame processing (so you can display the frames with detections)
def for_frame(frame_number, output_array, output_count, returned_frame):
    # Show the frame with detections using cv2.imshow()
    cv2.imshow("Detected Objects", returned_frame)

    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping the video...")
        camera.release()  # Release the camera feed
        cv2.destroyAllWindows()  # Close any OpenCV windows
        return False  # Return False to stop processing further frames
    print("------------END OF A FRAME --------------")
    return True

# Perform object detection on the video stream
detector.detectObjectsFromVideo(
    camera_input=camera,  # Input video stream (camera feed)
    output_file_path=output,  # Output video file
    frames_per_second=20,  # FPS
    minimum_percentage_probability=30,  # Minimum detection probability
    per_frame_function=for_frame,  # Callback function to process each frame
    return_detected_frame=True,
    detection_timeout=30
)

camera.release()
cv2.destroyAllWindows()