from ultralytics import YOLO
import cv2, math, numpy as np
import time


class Visualizer:
    def __init__(self):
        pass

    def readClassesFromFile(self, classesFilePath):
        '''Reads the class names from file, creates a color list for bounding boxes. Returns class names list and class colors.\n @param classFilePath Path of the class names file.'''
        np.random.seed(449)
        with open(classesFilePath, "r") as file:
            self.classes = file.read().splitlines()

            # Colors list for bounding box
            self.classesColors = np.random.uniform(low=0, high=255, size=(len(self.classes), 3))
        return self.classes, self.classesColors
    
    def createBoundingBox(self, img, bboxCoordinates, rectangleThickness=2, cornerThickness=3, bboxColor=(255,0,255), cornerColor=(0,255,0)):
        '''Creates a bounding box around detected class. It also adds diffrent colored corners.\n @param img Image.\n @param bboxCoordinates ((x min, y min), (x max, y max)) location of the bounding box.\n @param rectangleThickness Thickness of the bounding box.\n @param cornerThickness Thickness of the corners.\n @param bboxColor Color of the bounding box.\n @param cornerColor Color of corners.'''
        cv2.rectangle(img, (bboxCoordinates[0][0], bboxCoordinates[0][1]), (bboxCoordinates[1][0], bboxCoordinates[1][1]), bboxColor, rectangleThickness)

        line_width = min(int((bboxCoordinates[1][0] - bboxCoordinates[0][0]) * 0.2), int((bboxCoordinates[1][1] - bboxCoordinates[0][1])* 0.2))

        cv2.line(img, (bboxCoordinates[0][0], bboxCoordinates[0][1]), (bboxCoordinates[0][0] + line_width, bboxCoordinates[0][1]), cornerColor, cornerThickness)
        cv2.line(img, (bboxCoordinates[0][0], bboxCoordinates[0][1]), (bboxCoordinates[0][0], bboxCoordinates[0][1] + line_width), cornerColor, cornerThickness)

        cv2.line(img, (bboxCoordinates[0][0], bboxCoordinates[1][1]), (bboxCoordinates[0][0], bboxCoordinates[1][1] - line_width), cornerColor, cornerThickness)
        cv2.line(img, (bboxCoordinates[0][0], bboxCoordinates[1][1]), (bboxCoordinates[0][0] + line_width, bboxCoordinates[1][1]), cornerColor, cornerThickness)

        cv2.line(img, (bboxCoordinates[1][0], bboxCoordinates[0][1]), (bboxCoordinates[1][0], bboxCoordinates[0][1] + line_width), cornerColor, cornerThickness)
        cv2.line(img, (bboxCoordinates[1][0], bboxCoordinates[0][1]), (bboxCoordinates[1][0] - line_width, bboxCoordinates[0][1]), cornerColor, cornerThickness)

        cv2.line(img, (bboxCoordinates[1][0], bboxCoordinates[1][1]), (bboxCoordinates[1][0], bboxCoordinates[1][1] - line_width), cornerColor, cornerThickness)
        cv2.line(img, (bboxCoordinates[1][0], bboxCoordinates[1][1]), (bboxCoordinates[1][0] - line_width, bboxCoordinates[1][1]), cornerColor, cornerThickness)

    def putTextBoundingBox(self, img, bboxCoordinates, text="BBox Text: 1.0", textColor=(255,0,0), font=cv2.FONT_HERSHEY_PLAIN, 
                           fontSize=1, textThickness=2, recThickness=-1, recColor=(0,0,255)):
        '''Adds the text to the left top corner of detected class's bounding box. It automaticlly adjust the size of the textbox size.\n @param img Image.\n @param bboxCoordinates ((x min, y min), (x max, y max)) location of the bounding box.\n @param text Text.\n @param textColor Text color.\n @param font Text font style.\n @param fontSize Font size'''
        size, _ = cv2.getTextSize(text, font, fontSize, textThickness)
        textWidth, textHeight = size

        cv2.rectangle(img, (max(0, bboxCoordinates[0][0]), max(0, bboxCoordinates[0][1])), 
                      (max(0, bboxCoordinates[0][0]+textWidth), max(40, bboxCoordinates[0][1]-(textHeight*2))), recColor, recThickness)

        cv2.putText(img, text, (bboxCoordinates[0][0], bboxCoordinates[0][1]-int(textHeight/2)), font, fontSize, textColor, textThickness)
        
    def fpsCounter(self, img, cTime, pTime, pt=(20,30), color=(255,0,0)):
        '''@brief Shows the fps on the screen\n @param img Image.\n @param cTime Current time.\n @param pTime Previous time.\n @param pt Location of the fps counter on the screen.\n @param color Color'''
        fps = int(1/(cTime-pTime))
        cv2.putText(img, f"FPS: {fps}", pt, cv2.FONT_HERSHEY_PLAIN, 2, color, 2)