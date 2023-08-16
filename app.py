from TEKTech_Visual import *
import warnings, time

detector = Visualizer()
classeNames, classColors = detector.readClassesFromFile("classNames.names")

modelName = "yolov8l"
videoSource = "test/traffic.mp4"
model = YOLO("yolo_weights/"+modelName+".pt")

def predictImage(imagePath):
        image = cv2.imread(imagePath)
        result = model(image, show=True)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
def predictVideo(source, flipWebcam=True):
    'Start predicting in videos'
    cap = cv2.VideoCapture(source)
    cap.set(3, 1280)
    cap.set(4, 720)

    previousTime = 0

    while True:
        success, frame = cap.read()
        currentTime = time.time()
        if success:
            if source == 0 & flipWebcam == True:
                frame = cv2.flip(frame, 1)
            results = model(frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0]
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    bboxCoordinates = ((x_min, y_min), (x_max, y_max))

                    conf = math.floor(box.conf[0]*100)/100
                    classIndex = int(box.cls[0])

                    detector.createBoundingBox(frame, bboxCoordinates, bboxColor=classColors[classIndex])
                    detector.putTextBoundingBox(frame, bboxCoordinates, f"{classeNames[classIndex]}: {conf}")
            
            detector.fpsCounter(frame, currentTime, previousTime)
            previousTime = currentTime
            
            cv2.imshow("Result", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            warnings.warn("Video couldn't open. Try another source!", RuntimeWarning)
    cap.release()
    cv2.destroyAllWindows()


#predictImage("test/car.jpg")
predictVideo(videoSource)