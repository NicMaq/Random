import bpy
from bpy import context
import numpy as np
import os
import platform
import logging
import time
import threading
import mathutils
from typing import NamedTuple, Dict, Any
from datetime import datetime
import queue

print('platform is ',platform.system())
if platform.system() == 'Linux':
    print('Adding path')
    import site
    site.addsitedir('/home/nicolas/Workspace/ml/env310/lib/python3.10/site-packages')

import imageio
from skimage.transform import rescale, resize, downscale_local_mean
import cv2

from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface, SimulatorState, SimulatorSessionResponse


log = logging.getLogger("blender_simulator")
log.setLevel(logging.INFO)

workspace = os.getenv("SIM_WORKSPACE")
accesskey = os.getenv("SIM_ACCESS_KEY")

config_client = BonsaiClientConfig()
client = BonsaiClient(config_client)

registration_info = SimulatorInterface(
    name="Blender",
    timeout=60,
    simulator_context=config_client.simulator_context,
    description=None,
)

registered_session: SimulatorSessionResponse = client.session.create(workspace_name=config_client.workspace, body=registration_info)
print(f"Registered simulator. {registered_session.session_id}")


#>>> platform.system()
#Linux Darwin(MacOS) Windows
if platform.system() == 'Linux':
    log.info("Using Linux path")
    SAVE_DIR = '/home/nicolas/Workspace/ml/logs/FallingBalls' #Ubuntu
elif platform.system() == 'Darwin':
    log.info("Using MacOS path")
    SAVE_DIR = '/Users/nicolas/Workspace/ml/logs/FallingBalls' #MacOS
else:
    log.info("Using Windows path")
    SAVE_DIR = 'C:/Users/nmaquaire/Workspace/ml/logs/FallingBalls' #Windows

#Parameters
nBalls = 1
run_thread = True

class SimulatorModel:

    def __init__(self, q_2blender, q_2bonsai ):

        self.q_2blender = q_2blender
        self.q_2bonsai = q_2bonsai
        self.missedBalls = 0
        
    def reset(self, config) -> Dict[str, Any]:

        # Put message in q_2blender to ask Blender to reset
        log.info("\n\nq_2blender - Put Reset_Blender")
        msg2Blender = ('Reset_Blender', {'nBalls':1})
        self.q_2blender.put(msg2Blender)
        self.q_2blender.join()  

        # Get message from q_2bonsai to set returned states
        log.info("q_2bonsai - Get Reset_Bonsai")
        msgFromBlender = self.q_2bonsai.get()
        log.info(f'{"msgFromBlender is: {} ".format(msgFromBlender)}')
        states = msgFromBlender[1]['states']
        self.q_2bonsai.task_done()

        # return states
        returned_dict = self._gym_to_state(states, 0.0, False)
        
        return returned_dict


    def step(self, action) -> Dict[str, Any]:
        

        #log.info(f'{"step with action: {} ".format(action)}')

        # Put message in q_2blender to ask Blender to do a step
        log.info("\n\nq_2blender - Put Step_Blender")

        msg2Blender = ('Step_Blender', action)
        self.q_2blender.put(msg2Blender)
        self.q_2blender.join()  

        # Get message from q_2bonsai and set next_state, step_reward, done
        log.info("q_2bonsai - Get Step_Bonsai")
        msgFromBlender = self.q_2bonsai.get()
        log.info(f'{"msgFromBlender is: {} ".format(msgFromBlender)}')
        dataBlender = msgFromBlender[1]
        self.q_2bonsai.task_done()
        log.info("q_2bonsai - End Get Step_Bonsai")
        new_state = dataBlender['new_state']
        step_reward = dataBlender['step_reward']
        done = dataBlender['done']

        returned_dict = self._gym_to_state(new_state, step_reward, done) 

        return returned_dict


    def _gym_to_state(self, next_state, step_reward, done ):
        state = {
            "cart_pos": float(next_state[0]),
            "ball_y": float(next_state[1]),
            "ball_z": float(next_state[2]),
            "_gym_reward": float(step_reward),
            "_gym_terminal": float(done)    
        }

        return state       


def thread_simulate(q_2blender, q_2bonsai):

    global run_thread

    sequence_id = 1
    sim_model = SimulatorModel(q_2blender, q_2bonsai)
    sim_model_state = { 'sim_halted': False }

    try:
        while run_thread:
            sim_state = SimulatorState(sequence_id=sequence_id, state=sim_model_state, halted=sim_model_state.get('sim_halted', False))
            if sequence_id % 100 == 0: print('sim_state:', sim_state)

            event = client.session.advance(
                workspace_name=config_client.workspace,
                session_id=registered_session.session_id,
                body=sim_state,
            )
            sequence_id = event.sequence_id

            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
            elif event.type == "EpisodeStart":
                sim_model_state = sim_model.reset(event.episode_start.config)
            elif event.type == "EpisodeStep":
                sim_model_state = sim_model.step(event.episode_step.action)
            elif event.type == "EpisodeFinish":
                sim_model_state = { 'sim_halted': False }
            elif event.type == "Unregister":
                log.info(f'{"Simulator Session unregistered by platform because {} ".format(event.unregister.details)}')
                return

    except BaseException as err:
        client.session.delete(workspace_name=config_client.workspace, session_id=registered_session.session_id)
        log.info(f'{"Unregistered simulator because {} ".format(err)}')


def _initScene():

    _printActiveAndSelectedObjects('start init')

    for window in context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                #print(area.type)  PROPERTIES OUTLINER INFO OUTLINER TEXT_EDITOR VIEW_3D
                if area.type == 'VIEW_3D':
                    with context.temp_override(window=window, area=area):
                        
                        col = bpy.data.collections["Hidden"]
                        for obj in col.objects:
                            obj.data.materials.clear()        
                        
                        bpy.ops.object.select_all(action="SELECT")
                        obTCart = bpy.context.scene.objects["templateCart"]
                        obTCart.select_set(False)
                        obTSkull = bpy.context.scene.objects["templateSkull"]
                        obTSkull.select_set(False)                    
                        bpy.ops.object.delete(use_global=False, confirm=False)                     

                        # Add ground and back wall
                        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 0))
                        bpy.context.object.name = "Ground"
                        bpy.ops.transform.resize(value=(100, 100, 100), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=False, use_snap_edit=False, use_snap_nonedit=False, use_snap_selectable=False)
                        #bpy.ops.rigidbody.object_add()
                        #bpy.context.object.rigid_body.collision_shape = 'BOX'
                        #bpy.context.object.rigid_body.type = 'PASSIVE'

                        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 0))
                        bpy.context.object.name = "Sky"
                        bpy.ops.transform.resize(value=(100, 100, 100), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=False, use_snap_edit=False, use_snap_nonedit=False, use_snap_selectable=False)    
                        bpy.ops.transform.rotate(value=1.5708, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=False, use_snap_edit=False, use_snap_nonedit=False, use_snap_selectable=False)
                        bpy.ops.transform.translate(value=(10, 0, 0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=False, use_snap_edit=False, use_snap_nonedit=False, use_snap_selectable=False)

                        # Add cart (cube)
                        obTemplateCart = bpy.data.objects["templateCart"]
                        obCart = bpy.data.objects.new('Cart', obTemplateCart.data)
                        bpy.data.collections["Collection"].objects.link(obCart)
                        bpy.context.view_layer.objects.active = obCart   # Make the Cart the active object 
                        bpy.ops.rigidbody.object_add()
                        bpy.context.object.rigid_body.type = 'PASSIVE'
                        bpy.context.object.rigid_body.collision_shape = 'MESH'

                        # Add camera
                        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0), rotation=(1.11569, 0.0131949, -2.41231), scale=(1, 1, 1))
                        bpy.context.object.name = "Camera"
                        bpy.context.area.spaces.active.region_3d.view_perspective = 'CAMERA'
                        #bpy.ops.view3d.object_as_camera()
                        bpy.context.object.rotation_euler[2] = -1.5708
                        bpy.context.object.rotation_euler[0] = 1.5708
                        bpy.ops.transform.translate(value=(-50, -0, 6), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=False, use_snap_edit=False, use_snap_nonedit=False, use_snap_selectable=False)

                        # Add sun
                        bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(10, 0, 50), scale=(1, 1, 1))
                        
                        # Add new ball
                        obSkull = bpy.data.objects["templateSkull"]
                        obBall = bpy.data.objects.new('Ball', obSkull.data)
                        bpy.data.collections["Collection"].objects.link(obBall)
                        
                        y = np.random.randint(30, size=1)[0] - 15
                        obBall.location.x = 0
                        obBall.location.y = y
                        obBall.location.z = 16
                        bpy.context.view_layer.objects.active = obBall   # Make the Cart the active object 
                        bpy.ops.rigidbody.object_add()
                        bpy.context.object.rigid_body.collision_shape = 'MESH' 
                        
                        bpy.ops.object.select_all(action='DESELECT')  
                        bpy.context.view_layer.objects.active = None                                            
                        
                        break


    for window in context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                #print(area.type)  PROPERTIES OUTLINER INFO OUTLINER TEXT_EDITOR VIEW_3D
                if area.type == 'PROPERTIES':
                    with context.temp_override(window=window, area=area):
                        
                        #Delete all materials
                        for m in bpy.data.materials:
                            bpy.data.materials.remove(m)        
                            
                        # Create materials            
                        matCart = bpy.data.materials.new(name="MatCart") #set new material to variable
                        bpy.data.materials['MatCart'].diffuse_color = (0.8, 0.491085, 0.00577944, 1) #change color  

                        matGround = bpy.data.materials.new(name="MatGround") #set new material to variable
                        bpy.data.materials['MatGround'].diffuse_color = (0.00267726, 0.8, 0.0002749, 1) #change color  

                        matSky = bpy.data.materials.new(name="MatSky") #set new material to variable
                        bpy.data.materials['MatSky'].diffuse_color = (0.0029881, 0.477195, 0.8, 1) #change color  

                        matBall = bpy.data.materials.new(name="MatBall") #set new material to variable
                        bpy.data.materials['MatBall'].diffuse_color = (0.698773, 0.8, 0.759914, 1) #change color  

                        # rename data block and link material to object
                        obCart = bpy.context.scene.objects["Cart"]       # Get the object
                        obCart.data.name = "Cart"
                        bpy.context.view_layer.objects.active = obCart   # Make the Cart the active object 
                        obCart.data.materials.append(matCart) #add the material to the object
                                
                        obGround = bpy.context.scene.objects["Ground"]       # Get the object
                        obGround.data.name = "Ground"
                        bpy.context.view_layer.objects.active = obGround   # Make the Cart the active object 
                        obGround.data.materials.append(matGround) #add the material to the object

                        obSky = bpy.context.scene.objects["Sky"]       # Get the object
                        obSky.data.name = "Sky"
                        bpy.context.view_layer.objects.active = obSky   # Make the Cart the active object 
                        obSky.data.materials.append(matSky) #add the material to the object
                        
                        obBall = bpy.context.scene.objects["Ball"]       # Get the object
                        obBall.data.name = "Ball"    
                        obBall.data.materials.append(matBall) #add the material to the object   
                        
                        bpy.ops.object.select_all(action='DESELECT')  
                        bpy.context.view_layer.objects.active = None                         
                        
                        break

    _printActiveAndSelectedObjects('end init')                    


def _reset(config):

    _printActiveAndSelectedObjects('start reset')
              
    # Get the objects
    objectBalls = bpy.data.objects["Ball"]
    objectCart = bpy.data.objects["Cart"]

    # Reset Ball location
    y = np.random.randint(30, size=1)[0] - 15
    objectBalls.location = (0, y, 16)

    # Reset Cart location
    objectCart.location = location=(0, 0, 1)

    # Get locations
    coordinatesBall = objectBalls.location
    locBall_y = coordinatesBall[1]
    locBall_z= coordinatesBall[2]
    print('Reset coordinatesBall', coordinatesBall)
    #print('Reset locBall_y', locBall_y)
    #print('Reset locBall_z', locBall_z)
    coordinatesCart = objectCart.location  
    locCart = coordinatesCart[1]   
    print('Reset coordinatesCart', coordinatesCart)       
    #print('Reset locCart', locCart)

    new_state = [locCart, locBall_y, locBall_z] 

    _printActiveAndSelectedObjects('end reset')

    return new_state

def _step(actions, missedBalls):

        _printActiveAndSelectedObjects('start step')

        step_reward = 0
        done = False
        collided = False

        if actions['action'] == 0: move = mathutils.Vector([0.0, -1.0, 0.0])
        elif actions['action'] == 1: move = mathutils.Vector([0.0, 0.0, 0.0])
        else: move = mathutils.Vector([0.0, 1.0, 0.0])


        # Get the objects
        objectBalls = bpy.data.objects["Ball"] 
        objectCart = bpy.data.objects["Cart"]       # Get the object
        objectCart.location = objectCart.location + move
        print('Step objectCart new location is:', objectCart.location)

        # Get the locations
        coordinatesBall = objectBalls.matrix_world.to_translation()
        locBall_y = coordinatesBall[1]
        locBall_z= coordinatesBall[2]
        print('Step coordinatesBall', coordinatesBall)
        print('Step locBall_y', locBall_y)
        print('Step locBall_z', locBall_z)
        coordinatesCart = objectCart.location
        locCart = coordinatesCart[1]   
        print('Step coordinatesCart', coordinatesCart)       
        print('Step locCart', locCart)

        if _isCartCollision(coordinatesBall,coordinatesCart): 
            step_reward = 1
            _launchBalls()
            coordinatesBall = objectBalls.matrix_world.to_translation()
            locBall_y = coordinatesBall[1]
            locBall_z= coordinatesBall[2]
            collided = True

        if _isGroundCollision(coordinatesBall): 
            step_reward = -1  
            _launchBalls()
            coordinatesBall = objectBalls.matrix_world.to_translation()
            locBall_y = coordinatesBall[1]
            locBall_z= coordinatesBall[2]            
            collided = True
            missedBalls += 1
            if missedBalls >= 3:
                done = True       

        new_state = [locCart, locBall_y, locBall_z]
        print('Step new_state', new_state)

        info = {'missedBalls': missedBalls, 'collided': collided}

        _printActiveAndSelectedObjects('end step')

        return new_state, step_reward, done, info


def _launchBalls():

    _printActiveAndSelectedObjects('start launch')

    # Get the objects
    objectBalls = bpy.data.objects["Ball"]

    # Reset Ball location
    y = np.random.randint(30, size=1)[0] - 15
    objectBalls.location = (0, y, 16)

    _printActiveAndSelectedObjects('end launch')


def _isCartCollision(loc, locCart):

    collided = False
    
    if loc[1] <= (locCart[1] + 3) and loc[1] >= (locCart[1] - 3) and loc[2] <= (locCart[2]+ 1.5) :
        collided = True


    log.info(f'{"_isCartCollision is {} - locCart[1] is {} - locCart[2] is {}".format(collided,locCart[1] , locCart[2])}')
    log.info(f'{"loc[1] is {} - loc[2] is {}".format(loc[1],loc[2])}')

    return collided


def _isGroundCollision(loc):

    collided = False
    
    if (loc[2] - 0.5) <= 0:
        collided = True

    log.info(f'{"_isGroundCollision is: {} ".format(collided)}')

    return collided

def thread_gif(path, frames):

    _generate_gif(path,frames)


def _generate_gif(path_name, frames):

    for idx, frame_idx in enumerate(frames): 
        frames[idx] = resize(frame_idx, (576, 1024,3), preserve_range=True, order=0).astype(np.uint8)
        #frames[idx] = frame_idx.astype(np.uint8)

    imageio.mimsave(path_name, frames, duration=1/30)   


def _printActiveAndSelectedObjects(where):   

    selected = bpy.context.selected_objects
    print(f'{"{} selected objects are: {}".format(where, selected)}')

    activated = bpy.context.active_object
    print(f'{"{} active objects are: {}".format(where, activated)}')


if __name__ == '__main__':

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    
    # Instantiate the Queue object to manage all actions on BPY
    q_2blender = queue.Queue()
    q_2bonsai = queue.Queue()

    missedBalls = 0
    truncated = False
    done =False
    episode_reward = 0
    max_reward = -10
    frames = []
    frames_gif = []
    path_name = ""

    log.info("Thread simulate starting")
    #print('Thread simulate starting')
    threadSimulate = threading.Thread(target=thread_simulate, args=(q_2blender,q_2bonsai,))
    threadSimulate.start()

    # prepare a scene
    scn = bpy.context.scene
    bpy.context.window.scene = scn
    scn.frame_start = 0
    scn.frame_end = 70

    # Set first frame
    currentFrame = scn.frame_start
    scn.frame_set(currentFrame)

    simulationFile = "{}/simuBlender.png".format(SAVE_DIR)
    transparentFile = "{}/transparent.png".format(SAVE_DIR)
    scn.render.filepath = simulationFile    

    _initScene()

    try:

        while run_thread:

            if q_2blender.empty() is False:

                log.info("\n\nq_2blender - Get msgFromBonsai")
                msgFromBonsai = q_2blender.get()
                log.info(f'{"q_2blender - msgFromBonsai is: {} ".format(msgFromBonsai)}')

                if msgFromBonsai[0] == 'Reset_Blender':

                    log.info("\n\nStart Reset_Blender")
                    config = msgFromBonsai[1]
                    log.info(f'{"config is: {} ".format(config)}')

                    currentFrame = scn.frame_start
                    scn.frame_set(currentFrame)
                    
                    missedBalls = 0
                    truncated = False
                    done = False
                    episode_reward = 0
                    frames = []
                    states = _reset(config)  

                    # Save the render
                    bpy.ops.render.render( write_still=True )
                    image = cv2.imread(simulationFile)
                    frames.append(image.copy())

                    q_2blender.task_done()

                    # Add message in q_2bonsai with states
                    log.info("q_2bonsai - Put Reset_Bonsai")
                    msg2Bonsai = ('Reset_Bonsai', {'states':states})
                    q_2bonsai.put(msg2Bonsai)
                    log.info("q_2bonsai - End Put Reset_Bonsai")
                    q_2bonsai.join()    
                    log.info("End Reset_Blender")  
                
                if msgFromBonsai[0] == 'Step_Blender':

                    currentFrame += 1
                    scn.frame_set(currentFrame)

                    log.info("\n\nStart Step_Blender")
                    actions = msgFromBonsai[1]
                    log.info(f'{"actions is: {} ".format(actions)}')
                    log.info(f'{"currentFrame is: {} ".format(currentFrame)}')

                    new_state, step_reward, done, info = _step(actions, missedBalls)
                    episode_reward += step_reward

                    # Save the render
                    bpy.ops.render.render( write_still=True )
                    image = cv2.imread(simulationFile)
                    image = np.flip(image, axis=2)

                    #Add text
                    b,g,r,a = 255,0,0,0
                    text = "Team Bonsai: %s" % (episode_reward) 
                    image = np.ascontiguousarray(image, dtype=np.uint8)
                    cv2.putText(image,  text, (20,530), cv2.FONT_HERSHEY_SIMPLEX, 1, (b,g,r), 1, cv2.LINE_AA)

                    frames.append(image.copy())

                    if info['collided']:
                        currentFrame = scn.frame_start
                        scn.frame_set(currentFrame)                        
                        log.info(f'{"No increment as currentFrame is set to: {} ".format(currentFrame)}')  
                        log.info("info[missedBalls] is %s and missedBalls is %s" % (str(info['missedBalls']),str(missedBalls))) 
                        log.info("info[collided] is %s" % (str(info['collided'])))             
                    
                    if info['missedBalls'] > missedBalls: 
                        missedBalls = info['missedBalls']
                        log.info("max_reward is %s and episode_reward is %s" % (str(max_reward),str(episode_reward))) 
                        if missedBalls == 3: 
                            if max_reward <= episode_reward:
                                max_reward = episode_reward
                                path_name = SAVE_DIR + '/' + now + '_Blender_Balls_' + str(episode_reward) + '.gif'
                                frames_gif = frames.copy()
                                threadGif = threading.Thread(target=thread_gif, args=(path_name,frames_gif,))
                                threadGif.start()

                    if episode_reward >= 8 and max_reward == -10: 

                        path_name = SAVE_DIR + '/' + now + '_Blender_Balls_' + str(episode_reward) + '.gif'
                        frames_gif = frames.copy()
                        threadGif = threading.Thread(target=thread_gif, args=(path_name,frames_gif,))
                        threadGif.start()                                 

                    if currentFrame > scn.frame_end: 
                        truncated = True 
                        done = True #As truncated is not implemented
                    
                    q_2blender.task_done()                      

                    # Add message in q_2bonsai with new_state, step_reward, done and truncated
                    log.info("episode reward is %s" % (str(episode_reward))) 
                    log.info("q_2bonsai - Put Step_Bonsai")
                    msg2Bonsai = ('Step_Bonsai', {'new_state':new_state, 'step_reward':step_reward, 'done':done, 'truncated':truncated  })
                    q_2bonsai.put(msg2Bonsai) 
                    log.info("q_2bonsai - End Put Step_Bonsai")
                    q_2bonsai.join()
                    log.info("End Step_Blender")
                  
                    
    except KeyboardInterrupt:

        run_thread = False
        print('Process interupted')
        exit()

        raise




        






