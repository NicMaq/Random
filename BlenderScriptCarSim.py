import bpy
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import pickle
import threading
import time
import tensorflow as tf
import bpy
import sys
import os

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir )
    print(sys.path)
#import env 

SAVE_DIR = '/Users/nicolasmaquaire/Dropbox/Model.fit/Products/200731_SimuBlender'

#Parameters
heightFrame = 80
deltaSlopes = 0.5
slopeMax = 1000
heading = 90
motion = 0
#currentFrame = 1

camera_matrix = np.matrix([[513.15872345, 0.0, 318.47635449],[0.0, 514.02757839, 241.76495883],[0.0, 0.0, 1.0]])
dist_coefs = np.matrix([0.205176489, -0.658394196, -0.0000385006371, -0.000827979084, 0.592718126])

perspective_matrix = np.matrix([[-1.00971472e+00,-1.32601090e+00,6.40901213e+02],[ 0.00000000e+00,-2.87165963e+00,8.95957805e+02],[ 2.89136194e-06,-4.16974030e-03,1.00000000e+00]])
perspective_matrix_inv = np.matrix([[ 2.98063371e-01,-4.64340143e-01, 2.25000000e+02],[ 8.93425545e-04,-3.48869774e-01,3.12000000e+02],[ 2.86354341e-06,-1.45335378e-03,1.00000000e+00]])


def thread_getHeadingAndMotion(newFrame, nFrame):
    print('Thread getHeadingAndMotion starting')

    threadObjectDetection = threading.Thread(target=thread_objectDetection, args=(newFrame,))
    threadObjectDetection.start()

    threadLinesDetection = threading.Thread(target=thread_linesDetection, args=(newFrame, nFrame))
    threadLinesDetection.start()

    threadObjectDetection.join()
    threadLinesDetection.join()

    print('Thread getHeadingAndMotion finishing')

def thread_objectDetection(simulationFile):
    print('Thread objectDetection starting')
    time.sleep(2)
    print('Thread objectDetection finishing')
    
def thread_linesDetection(image, nFrame):
    print('Thread linesDetection starting')

    lowerFrame = lineDetection(image)
    nameOutput = "{}/lowerFrame{}.png".format(SAVE_DIR, nFrame)
    #nameOutput = "/Users/nicolasmaquaire/Dropbox/HomeOffice/Multimedia/190511_Car/Simulation/lowerFrame%s.png" % nFrame
    cv2.imwrite(nameOutput,lowerFrame)

    print('Thread linesDetection finishing')

def main():
    # prepare a scene
    #global currentFrame
    scn = bpy.context.scene
    scn.frame_start = 1
    scn.frame_end = 500
    
    #Get blender objects and positions for the two first frames
    currentFrame = scn.frame_start
    scn.frame_set(currentFrame)
    objectCar = bpy.data.objects["Car"]
    objectArmature = bpy.data.objects["Armature"]
    objectRoad = bpy.data.objects["Road"]
    #loc, _, _ = objectCar.matrix_world.decompose()
    loc = objectCar.matrix_world.to_translation()
    rot_euler = objectCar.matrix_world.to_euler('XYZ')
    rot_euler_d = math.degrees(rot_euler[2])

    currentFrame+=1
    scn.frame_set(currentFrame)
    newLoc = objectCar.matrix_world.to_translation()
    newRot_euler = objectCar.matrix_world.to_euler('XYZ')
    newRot_euler_d = math.degrees(newRot_euler[2])
    
    #Store data and images
    #simulationFile = '/Users/nicolasmaquaire/Dropbox/HomeOffice/Multimedia/190511_Car/Simulation/simuBlender.png'
    #simulationFile = '/home/nicolas/Dropbox/HomeOffice/Multimedia/190511_Car/Simulation/simuBlender.png'
    simulationFile = "{}/simuBlender.png".format(SAVE_DIR)
    scn.render.filepath = simulationFile
    driveDataSet = []
    #DatasetName = "/Users/nicolasmaquaire/Dropbox/HomeOffice/Multimedia/190511_Car/Simulation/drivingLabels"
    #DatasetName = "/home/nicolas/Dropbox/HomeOffice/Multimedia/190511_Car/Simulation/drivingLabels"
    DatasetName = "{}/Simulation/drivingLabels".format(SAVE_DIR)
    
    while currentFrame < scn.frame_end: #< 4: 
        
        print("render frame:", currentFrame)
        bpy.ops.render.render( write_still=True ) 
        
        #Generate Labels Motion and Heading with displacement (occured in 1/24s 1/framerate)
        dispX = newLoc[0] - loc[0]
        dispY = newLoc[1] - loc[1]
        deltaRotation = newRot_euler_d - rot_euler_d
        deltaRotation = 180 if deltaRotation>= 180 else deltaRotation
        
        motion = math.sqrt(dispX**2+dispY**2)*24
        print('motion',motion)
        
        
        heading = deltaRotation *24
        print('heading',heading)
        
        driveDataSet.append((heading,motion))
        
        loc = newLoc
        rot_euler_d = newRot_euler_d
        
        #Calculate reward function
        #isOnTrack = 10 if onTrack() else 0
        #print('isOnTrack',isOnTrack)
        #isTopSpeed = motion
        
        #reward = isOnTrack + isTopSpeed
        #rint('reward',reward)

        # Overlay Image
        #overlayImage = Image.new("RGB",(640, 240))
        # get a font
        #fnt = font = ImageFont.load_default() #truetype("arial.ttf", 15) #ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 15)

        #imgOverlay = overlayImage.copy()

        #image = cv2.imread(simulationFile)

        #print('new render\'s shape is',image.shape)
        #image_udst = undistortCam(image)
        #print('render_undistorted',image_udst.shape)   

        #print('Thread getHeadingAndMotion starting for frame %s' % currentFrame)
        #threadGetHeadingAndMotion = threading.Thread(target=thread_getHeadingAndMotion, args=(image_udst,currentFrame,))
        #threadGetHeadingAndMotion.start()
           
        # Detect Objects with the 1st DNN
        #imgOverlay, objectCoordinates = objectDetection(engine, labels, fnt, image_udst, imgOverlay)

        currentFrame+=1
        scn.frame_set(currentFrame)
        newLoc = objectCar.matrix_world.to_translation()
        newRot_euler = objectCar.matrix_world.to_euler('XYZ')
        newRot_euler_d = math.degrees(newRot_euler[2])
        
        print('newLoc',newLoc)
        print('loc',loc)
        print('newRot_euler',newRot_euler)
        print('rot_euler',rot_euler)
    
    
    # open the file for writing
    fileObject = open(DatasetName,'wb') 
    pickle.dump(driveDataSet,fileObject)
    fileObject.close()
        



        
        
if __name__ == '__main__':
    main()

