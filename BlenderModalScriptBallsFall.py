import bpy
from bpy import context

import numpy as np
import os
import logging
import time
import threading
# Manage communication between threads
# msg2Blender = ('Reset_Blender', {'nBalls':1})

import queue

import site
site.addsitedir('/home/nicolas/Workspace/ml/env310/lib/python3.10/site-packages')

from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface, SimulatorState, SimulatorSessionResponse
from typing import NamedTuple, Dict, Any

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


SAVE_DIR = '/home/nicolas/Workspace/ml/logs/FallingBalls'

#Parameters
heightFrame = 80
widthFrame = 80
nBalls = 1

class SimulatorModel:

    def __init__(self, q_2blender, q_2bonsai ):

        self.q_2blender = q_2blender
        self.q_2bonsai = q_2bonsai
        self.missedBalls = 0
        
    def reset(self, config) -> Dict[str, Any]:

        # Put message in q_2blender to ask Blender to reset
        print('q_2blender - Put Reset_Blender')
        msg2Blender = ('Reset_Blender', {'nBalls':1})
        self.q_2blender.put(msg2Blender)
        print('q_2blender - End Put Reset_Blender')
        self.q_2blender.join()  

        # Get message from q_2bonsai to set returned states
        print('q_2bonsai - Get Reset_Bonsai')
        msgFromBlender = self.q_2bonsai.get()
        print('msgFromBlender is: ', msgFromBlender)
        states = msgFromBlender[1]['states']
        self.q_2bonsai.task_done()
        print('q_2bonsai - End Get Reset_Bonsai')

        # return states
        returned_dict = self._gym_to_state(states, 0.0, False)
        
        return { 
            'sim_halted': False,
            'key': returned_dict
        }


    def step(self, action) -> Dict[str, Any]:
        
        #if self.args.debug: 
        print('step with action: ', action )
        # Add message in queue to step

        # Put message in q_2blender to ask Blender to do a step
        print('q_2blender - Put Step_Blender')
        msg2Blender = ('Step_Blender', {'actions':action})
        self.q_2blender.put(msg2Blender)
        print('q_2blender - End Put Step_Blender')
        self.q_2blender.join()  

        # Get message from q_2bonsai and set next_state, step_reward, done
        print('q_2bonsai - Get Step_Bonsai')
        msgFromBlender = self.q_2bonsai.get()
        print('msgFromBlender is: ', msgFromBlender)
        listFromBLender = msgFromBlender[1]
        self.q_2bonsai.task_done()
        print('q_2bonsai - End Get Step_Bonsai')

        # Read next_state, step_reward, done, truncated, info        
        #next_state, step_reward, done, truncated, info = self._step(action['action']) # next_state, step_reward, done, info
        returned_dict = {} #self._gym_to_state(next_state, step_reward, done)
        #if self.args.debug: 
        #print('next_state is: ', next_state )
        #print('step_reward is: ', step_reward )

        return {
            'sim_halted': False,
            'key': returned_dict
        }   


    def _gym_to_state(self, next_state, step_reward, done ):
        state = {
            "cart_pos": float(next_state[0]),
            "ball_y": float(next_state[1]),
            "ball_z": float(next_state[2]),
            "_gym_reward": float(step_reward),
            "_gym_terminal": float(done)    
        }

        return state       

    
    '''
    def _preprocess(self, image):

        img_gray = tf.image.rgb_to_grayscale(image)

        if img_gray.shape[0] != heightFrame:
            img_resized = tf.image.resize(img_gray, [heightFrame, widthFrame], method='nearest')  
            return img_resized  

        return img_gray    
    '''


def thread_simulate(q_2blender, q_2bonsai):

    sequence_id = 1
    sim_model = SimulatorModel(q_2blender, q_2bonsai)
    sim_model_state = { 'sim_halted': False }

    try:
        while True:
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
                print(f"Simulator Session unregistered by platform because '{event.unregister.details}'")
                return

    except BaseException as err:
        client.session.delete(workspace_name=config_client.workspace, session_id=registered_session.session_id)
        print(f"Unregistered simulator because {type(err).__name__}: {err}")

def _step(actions, missedBalls):

        # Get the objects
        objectBalls = bpy.data.objects["Ball"]
        objectCart = bpy.data.objects["Cart"]

        # Move the cart with actions (left, right or nothing)
        # Set currentframe
        
        # Get locations
        coordinatesBall = objectBalls.matrix_world.to_translation()
        locBall_y = coordinatesBall[1]
        locBall_z= coordinatesBall[2]
        print('Reset coordinatesBall', coordinatesBall)
        print('Reset locBall_y', locBall_y)
        print('Reset locBall_z', locBall_z)
        coordinatesCart = objectCart.matrix_world.to_translation()  
        locCart = coordinatesCart[1]   
        print('Reset coordinatesCart', coordinatesCart)       
        print('Reset locCart', locCart)

        if _isCartCollision(coordinatesBall,coordinatesCart): 
            step_reward = 1
            _removeBall()
            _addBall()

        if _isGroundCollision(coordinatesBall): 
            step_reward = -1  
            _removeBall()
            missedBalls += 1
            if missedBalls >= 3:
                done = True       

        new_state = [locCart, locBall_y, locBall_z]
        info = {'missedBalls': missedBalls}

        return new_state, step_reward, done, info

def _reset(config):
     
        for window in context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                #print(area.type)  PROPERTIES OUTLINER INFO OUTLINER TEXT_EDITOR VIEW_3D
                if area.type == 'VIEW_3D':
                    with context.temp_override(window=window, area=area):
                        # Delete all
                        bpy.ops.object.select_all(action='SELECT')
                        bpy.ops.object.delete(use_global=False, confirm=False)
                        
                        # Add new ball
                        _addBall()
                        
                        # Add ground and back wall
                        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 0))
                        bpy.context.object.name = "Ground"
                        bpy.ops.transform.resize(value=(100, 100, 100), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=False, use_snap_edit=False, use_snap_nonedit=False, use_snap_selectable=False)
                        bpy.ops.rigidbody.object_add()
                        #bpy.context.object.rigid_body.collision_shape = 'BOX'
                        #bpy.context.object.rigid_body.type = 'PASSIVE'


                        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 0))
                        bpy.context.object.name = "Sky"
                        bpy.ops.transform.resize(value=(100, 100, 100), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=False, use_snap_edit=False, use_snap_nonedit=False, use_snap_selectable=False)    
                        bpy.ops.transform.rotate(value=1.5708, orient_axis='Y', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, True, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=False, use_snap_edit=False, use_snap_nonedit=False, use_snap_selectable=False)
                        bpy.ops.transform.translate(value=(10, 0, 0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=False, use_snap_edit=False, use_snap_nonedit=False, use_snap_selectable=False)
                    
                        # Add cart (cube)
                        bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 1), scale=(3, 3, 1))
                        bpy.context.object.name = "Cart"
                        #bpy.ops.rigidbody.object_add()
                        #bpy.context.object.rigid_body.collision_shape = 'BOX'
                    
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


                    break
                
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
                        bpy.data.materials['MatBall'].diffuse_color = (0.8, 0, 0.000847888, 1) #change color  


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
                        bpy.context.view_layer.objects.active = obBall   # Make the Cart the active object 
                        obBall.data.materials.append(matBall) #add the material to the object

                    break
                
        # Get the objects
        objectBalls = bpy.data.objects["Ball"]
        objectCart = bpy.data.objects["Cart"]
        
        # Get locations
        coordinatesBall = objectBalls.matrix_world.to_translation()
        locBall_y = coordinatesBall[1]
        locBall_z= coordinatesBall[2]
        print('Reset coordinatesBall', coordinatesBall)
        print('Reset locBall_y', locBall_y)
        print('Reset locBall_z', locBall_z)
        coordinatesCart = objectCart.matrix_world.to_translation()  
        locCart = coordinatesCart[1]   
        print('Reset coordinatesCart', coordinatesCart)       
        print('Reset locCart', locCart)
        
        new_state = [locCart, locBall_y, locBall_z]
        return new_state


def _addBall():

    for window in context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            #print(area.type)  PROPERTIES OUTLINER INFO OUTLINER TEXT_EDITOR VIEW_3D
            if area.type == 'VIEW_3D':
    
                # Randomize position
                y = np.random.randint(20, size=1)[0]
                bpy.ops.mesh.primitive_uv_sphere_add(enter_editmode=False, align='WORLD', location=(0, y, 16), scale=(1, 1, 1))
                bpy.context.object.name = "Ball"
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.collision_shape = 'SPHERE'


def _removeBall():
    
    # Delete objects
    objectBalls = bpy.data.objects["Ball"]  
    objectBalls.delete()


def _isCartCollision(self, loc, locCart):

    collided = False
    
    if loc[1] <= (locCart[1] + 1) and loc[1] >= (locCart[1] - 1) and (loc[2] - 0.5) <= locCart[2] :
        collided = True

    return collided


def _isGroundCollision(loc):

    collided = False
    
    if (loc[2] - 0.5) <= 0:
        collided = True

    return collided



class BonsaiConnector(bpy.types.Operator):
    bl_idname = "object.modal_bonsai_connector"
    bl_label = "Bonsai Connector"

    def __init__(self):
        print("Start")
        # Instantiate the Queue object to manage all actions on BPY
        self.q_2blender = queue.Queue()
        self.q_2bonsai = queue.Queue()

        print('Thread simulate starting')
        threadSimulate = threading.Thread(target=thread_simulate, args=(self.q_2blender,self.q_2bonsai,))
        threadSimulate.start()

        # prepare a scene
        self.scn = bpy.context.scene
        self.scn.frame_start = 1
        self.scn.frame_end = 500

        # Set first frame
        currentFrame = self.scn.frame_start
        self.scn.frame_set(currentFrame)

        simulationFile = "{}/simuBlender.png".format(SAVE_DIR)
        self.scn.render.filepath = simulationFile    

        self.connect_bonsai = False

    def __del__(self):
        print("End")

    def execute(self, context):
        while  self.connect_bonsai:

            if self.q_2blender.empty() is False:

                print('q_2blender - Get msgFromBonsai')
                msgFromBonsai = self.q_2blender.get()
                print('q_2blender - msgFromBonsai is: ',msgFromBonsai)

                if msgFromBonsai[0] == 'Reset_Blender':
                    print('Start Reset_Blender')
                    config = msgFromBonsai[1]
                    print('config is: ', config)
                    self.scn.frame_set(self.scn.frame_start)
                    states = _reset(config)  
                    self.q_2blender.task_done()

                    # Add message in q_2bonsai with states
                    print('q_2bonsai - Put Reset_Bonsai')
                    msg2Bonsai = ('Reset_Bonsai', {'states':states})
                    self.q_2bonsai.put(msg2Bonsai)
                    print('q_2bonsai - End Put Reset_Bonsai')
                    self.q_2bonsai.join()    
                    print('End Reset_Blender')          
                
                if msgFromBonsai[0] == 'Step_Blender':
                    actions = msgFromBonsai[1]
                    currentFrame += 1
                    self.scn.frame_set(currentFrame)
                    new_state, step_reward, done, info = _step(actions)
                    if currentFrame > self.scn.frame_end: 
                        truncated = True 

                    # Add message in queue to ask Blender to reset
                    print('Put Step_Bonsai')
                    msg2Bonsai = ('Step_Bonsai', {'new_state':states, 'step_reward':step_reward, 'done':done, 'truncated':truncated  })
                    self.q_bpy.put(msg2Bonsai)                       
                
        return {'FINISHED'}

    def modal(self, context, event):
        if event.type in {'ESC'}:  # Cancel
            self.connect_bonsai = False
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.connect_bonsai = True
        self.execute(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    



if __name__ == '__main__':

    bpy.utils.register_class(BonsaiConnector)

    # Invoke the connector 
    bpy.ops.object.modal_bonsai_connector('INVOKE_DEFAULT')









    






