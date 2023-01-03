#Used to connect to Ansys
import random
import socket
import struct
import time
import sys
import platform

#Used to load the model and launch the simualtion
import os
from pyaedt import TwinBuilder
from pyaedt import generate_unique_project_name

#Used by Bonsai
import logging
import time
import threading
import mathutils
from typing import NamedTuple, Dict, Any
from datetime import datetime
import queue
#import imageio
#from skimage.transform import rescale, resize, downscale_local_mean
#import cv2

from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface, SimulatorState, SimulatorSessionResponse


# Define host and port where the DataConnector socket is listening 
host = '127.0.0.1'
port = int(5010)
comm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
comm_socket.settimeout(10)

#Set logging
#log = logging.getLogger("ansys_simulator")
log = logging.getLogger()
log.setLevel(logging.INFO)

#Set Bonsai
workspace = os.getenv("SIM_WORKSPACE")
accesskey = os.getenv("SIM_ACCESS_KEY")

config_client = BonsaiClientConfig()
client = BonsaiClient(config_client)

registration_info = SimulatorInterface(
    name="Ansys",
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
    SAVE_DIR = '/home/nicolas/Workspace/ml/logs/Ansys' #Ubuntu
elif platform.system() == 'Darwin':
    log.info("Using MacOS path")
    SAVE_DIR = '/Users/nicolas/Workspace/ml/logs/Ansys' #MacOS
else:
    log.info("Using Windows path")
    SAVE_DIR = 'C:/Users/nmaquaire/Workspace/ml/logs/Ansys' #Windows

#Global parameters
run_thread = True    


class SimulatorModel:

    def __init__(self, q_2ansys, q_2bonsai ):

        self.q_2ansys = q_2ansys
        self.q_2bonsai = q_2bonsai

        
    def reset(self, config) -> Dict[str, Any]:

        # Put message in q_2ansys to ask Ansys to reset
        log.info("\n\nq_2ansys - Put Reset_Ansys")
        msg2Ansys = ('Reset_Ansys', {'h':2.2})
        self.q_2ansys.put(msg2Ansys)
        self.q_2ansys.join()  

        # Get message from q_2bonsai to set returned states
        log.info("q_2bonsai - Get Reset_Bonsai")
        msgFromAnsys = self.q_2bonsai.get()
        log.info(f'{"msgFromAnsys is: {} ".format(msgFromAnsys)}')
        states = msgFromAnsys[1]['states']
        self.q_2bonsai.task_done()

        # return states
        returned_dict = self._gym_to_state(states, 0.0, False)
        
        return returned_dict


    def step(self, action) -> Dict[str, Any]:
        

        #log.info(f'{"step with action: {} ".format(action)}')

        # Put message in q_2ansys to ask Ansys to do a step
        log.info("\n\nq_2ansys - Put Step_Ansys")

        msg2Ansys = ('Step_Ansys', action)
        self.q_2ansys.put(msg2Ansys)
        self.q_2ansys.join()  

        # Get message from q_2bonsai and set next_state, step_reward, done
        log.info("q_2bonsai - Get Step_Bonsai")
        msgFromAnsys = self.q_2bonsai.get()
        log.info(f'{"msgFromAnsys is: {} ".format(msgFromAnsys)}')
        dataAnsys = msgFromAnsys[1]
        self.q_2bonsai.task_done()
        log.info("q_2bonsai - End Get Step_Bonsai")
        new_state = dataAnsys['new_state']
        step_reward = dataAnsys['step_reward']
        done = dataAnsys['done']

        returned_dict = self._gym_to_state(new_state, step_reward, done) 

        return returned_dict


    def _gym_to_state(self, next_state, step_reward, done ):
        state = {
            "fromDataConnector1": float(next_state[0]),
            "fromDataConnector2": float(next_state[0]),
            "_gym_reward": float(step_reward),
            "_gym_terminal": float(done)    
        }

        return state       


def thread_simulate(q_2ansys, q_2bonsai):

    global run_thread

    sequence_id = 1
    sim_model = SimulatorModel(q_2ansys, q_2bonsai)
    sim_model_state = { 'sim_halted': False }

    try:
        config = {'h':2.2}
        _ = sim_model.reset(config)
        sim_model.step(0)
        '''
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
        '''
    except BaseException as err:
        client.session.delete(workspace_name=config_client.workspace, session_id=registered_session.session_id)
        log.info(f'{"Unregistered simulator because {} ".format(err)}')


def thread_analyze(tb):

        tb.close_project(save_project=False)
        tb.load_project('C:/Users/nmaquaire/Workspace/Ansys/Projects/PumpingSystem.aedt', design_name='PumpingDesign')
        tb.analyze_setup('TR')


def _initScene():

    desktop_version = "2022.2"

    non_graphical = os.getenv("PYAEDT_NON_GRAPHICAL", "False").lower() in ("true", "1", "t")
    new_thread = True
    tb = None
    
    try:
        #tb =  TwinBuilder(specified_version=desktop_version, non_graphical=False, new_desktop_session=new_thread)
        tb =  TwinBuilder(specified_version=desktop_version, non_graphical=False, new_desktop_session=False, aedt_process_idint=64320 )
    except BaseException as err:
        print('Error loading the Twin Builder project...')

    return tb

def _step(actions):

        step_reward = 0
        done = False
        truncated = False        
        info = {}        

        val1, val2, val3 = random.random(), random.random(),  random.random()
        print('Sent to Twin Builder: {},{}'.format(val1, val2, val3))

        # Data sent through the socket needs to be in binary format
        # '!dd' format indicates that 2 double-values will be packed
        #  into binary form using network byte order
        packet = struct.pack('!dd', val1, val2, val3)
        # data.encode()
        #packet = struct.pack('!dd', val1)

        try:
            comm_socket.sendall(packet)
            # '24' indicates that three 8-byte values will return from the simulatiom
            packet = comm_socket.recv(24) 
        except ConnectionAbortedError as e:
            print('Data socket connection lost. Shutting down...')
            sys.exit(-1)

        if not packet:
            pass #continue
        sim_data = struct.unpack('!ddd', packet)
        print('Received from Twin Builder: {}'.format(sim_data))        

        new_state = [sim_data[0], sim_data[1], sim_data[2]]
        print('Step new_state', new_state)

        return new_state, step_reward, done, truncated, info    

def _reset(tb, config):

        step_reward = 0
        done = False        
        truncated = False
        info = {}    

        log.info("Thread analyze starting")
        threadAnalyze = threading.Thread(target=thread_analyze, args=(tb,))
        threadAnalyze.start()

        time.sleep(3)

        # Connects to the socket
        try:
            comm_socket.connect((host, port))
        except ConnectionRefusedError:
            print('Could not find socket to connect. Make sure the simulation is in the Initialize state, and that the localhost and port number are correct')
            sys.exit(-1)

        #Then re-open
        new_state = [2.2,0,0]

        return new_state, step_reward, done, truncated, info  


if __name__ == '__main__':

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    
    # Instantiate the Queue object to manage all actions on BPY
    q_2ansys = queue.Queue()
    q_2bonsai = queue.Queue()

    simulation_samples_index = 0
    truncated = False
    done =False
    episode_reward = 0

    # prepare Twin Builder
    log.info("Twin Builder starting")
    tb = _initScene()    

    log.info("Thread simulate starting")
    threadSimulate = threading.Thread(target=thread_simulate, args=(q_2ansys,q_2bonsai,))
    threadSimulate.start()

    try:

        while run_thread:

            if q_2ansys.empty() is False:

                log.info("\n\nq_2ansys - Get msgFromBonsai")
                msgFromBonsai = q_2ansys.get()
                log.info(f'{"q_2ansys - msgFromBonsai is: {} ".format(msgFromBonsai)}')

                if msgFromBonsai[0] == 'Reset_Ansys':

                    log.info("\n\nStart Reset_Ansys")
                    config = msgFromBonsai[1]
                    log.info(f'{"config is: {} ".format(config)}')

                    new_state, step_reward, done, truncated, info = _reset(tb, config)
                    episode_reward = 0

                    simulation_samples_index = 0

                    q_2ansys.task_done()

                    # Add message in q_2bonsai with states
                    log.info("q_2bonsai - Put Reset_Bonsai")
                    msg2Bonsai = ('Reset_Bonsai', {'states':new_state})
                    q_2bonsai.put(msg2Bonsai)
                    log.info("q_2bonsai - End Put Reset_Bonsai")
                    q_2bonsai.join()    
                    log.info("End Reset_Ansys")  
                
                if msgFromBonsai[0] == 'Step_Ansys':

                    log.info("\n\nStart Step_Ansys")
                    actions = msgFromBonsai[1]
                    log.info(f'{"actions is: {} ".format(actions)}')
                    log.info(f'{"simulation_index is: {} ".format(simulation_samples_index)}')

                    new_state, step_reward, done, truncated, info = _step(actions)
                    episode_reward += step_reward

                    simulation_samples_index += 1

                    #if simIndex > scn.frame_end: 
                    #    truncated = False 
                    #    done = True #As truncated is not implemented
                    
                    q_2ansys.task_done()                      

                    # Add message in q_2bonsai with new_state, step_reward, done and truncated
                    log.info("episode reward is %s" % (str(episode_reward))) 
                    log.info("q_2bonsai - Put Step_Bonsai")
                    msg2Bonsai = ('Step_Bonsai', {'new_state':new_state, 'step_reward':step_reward, 'done':done, 'truncated':truncated  })
                    q_2bonsai.put(msg2Bonsai) 
                    log.info("q_2bonsai - End Put Step_Bonsai")
                    q_2bonsai.join()
                    log.info("End Step_Ansys")
                  
                    
    except KeyboardInterrupt:

        run_thread = False
        comm_socket.close()
        tb.release_desktop()
        print('Process interupted')
        exit()

        raise

    run_thread = False
    tb.release_desktop()
    comm_socket.close()
    exit()




        






