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
print('workspace is: ', workspace)

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
        
        global host, port, comm_socket

        self.q_2ansys = q_2ansys
        self.q_2bonsai = q_2bonsai

        
    def reset(self, config) -> Dict[str, Any]:

        # Raise the Ansys timeout
        log.info("\n\nSleep to timeout TR")
        time.sleep(6)

        # Put message in q_2ansys to ask Ansys to reset
        log.info("\nq_2ansys - Put Reset_Ansys")
        msg2Ansys = ('Relaunch', 'Relaunch Analyze TR')
        self.q_2ansys.put(msg2Ansys)
        self.q_2ansys.join()

        # Get message from q_2bonsai to set returned states
        #log.info("q_2bonsai - Get Reset_Bonsai")
        #msgFromAnsys = self.q_2bonsai.get()
        #log.info(f'{"msgFromAnsys is: {} ".format(msgFromAnsys)}')
        log.info("Set new states")
        states = [2.2, 0, 0]
        #self.q_2bonsai.task_done()

        # Wait for TR to initialize
        log.info("\n\nSleep to let TR launch")
        time.sleep(3)

        # Connects to the socket
        log.info("\n\nConnects to the socket")
        try:
            comm_socket.connect((host, port))
        except ConnectionRefusedError:
            print('Could not find socket to connect. Make sure the simulation is in the Initialize state, and that the localhost and port number are correct')
            sys.exit(-1)

        # return states
        returned_dict = self._gym_to_state(states, 0.0, False)
        
        return returned_dict


    def step(self, action) -> Dict[str, Any]:
        

        log.info(f'{"step with action: {} ".format(action)}')

        # Put message in q_2ansys to ask Ansys to do a step
        log.info("\n\nStep_Ansys")

        step_reward = 0
        done = False
        truncated = False        
        info = {}        

        val1, val2, val3 = random.random(), random.random(), random.random()
        print('Step - Sent to Twin Builder: {},{},{}'.format(val1, val2, val3))

        # Data sent through the socket needs to be in binary format
        # '!dd' format indicates that 2 double-values will be packed
        #  into binary form using network byte order
        packet = struct.pack('!ddd', val1, val2, val3)
        # data.encode()
        #packet = struct.pack('!dd', val1)

        try:
            comm_socket.sendall(packet)
            # '24' indicates that three 8-byte values will return from the simulatiom
            packet = comm_socket.recv(24) 
        except ConnectionAbortedError as e:
            print('Data socket connection lost. Shutting down...')
            sys.exit(-1)

        sim_data = struct.unpack('!ddd', packet)
        print('Received from Twin Builder: {}'.format(sim_data))        

        new_state = [sim_data[0], sim_data[1], sim_data[2]]
        print('Step new_state', new_state)

        #step_reward = dataAnsys['step_reward']
        #done = dataAnsys['done']

        returned_dict = self._gym_to_state(new_state, step_reward, done) 

        return returned_dict


    def _gym_to_state(self, new_state, step_reward, done ):
        dict_state = {
            "fromDataConnector1": float(new_state[0]),
            "fromDataConnector2": float(new_state[1]),
            "fromDataConnector3": float(new_state[2]),            
            "_gym_reward": float(step_reward),
            "_gym_terminal": float(done)    
        }

        return dict_state       


def thread_simulate(q_2ansys, q_2bonsai):

    global run_thread

    sequence_id = 1
    sim_model = SimulatorModel(q_2ansys, q_2bonsai)
    sim_model_state = { 'sim_halted': False }

    #Wait for Ansys to start Analyze
    time.sleep(1)

    '''
    try:

    config = {'h':2.2}
    print('Reset')
    _ = sim_model.reset(config)
    time.sleep(1)
    print('Step0')
    sim_model.step(0)
    time.sleep(1)
    print('Step1')
    sim_model.step(0)
    time.sleep(1)
    print('Step2')
    sim_model.step(0)
    time.sleep(1)
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
        
    except BaseException as err:
        client.session.delete(workspace_name=config_client.workspace, session_id=registered_session.session_id)
        log.info(f'{"Unregistered simulator because {} ".format(err)}')


def _initScene(tb):

    try:
        tb.close_project(save_project=False)
        tb.load_project('C:/Users/nmaquaire/Workspace/Ansys/Projects/PumpingSystem.aedt', design_name='PumpingDesign')
   
    except BaseException as err:
        print('Error loading the Twin Builder project...')


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
    desktop_version = "2022.2"
    non_graphical = os.getenv("PYAEDT_NON_GRAPHICAL", "False").lower() in ("true", "1", "t")
    new_thread = True
    log.info("Twin Builder starting")
    tb =  TwinBuilder(specified_version=desktop_version, non_graphical=False, new_desktop_session=new_thread)    

    _initScene(tb)
    
    #Start connection with Bonsai
    log.info("Thread simulate starting")
    threadSimulate = threading.Thread(target=thread_simulate, args=(q_2ansys,q_2bonsai,))
    threadSimulate.start()

    tb.analyze_setup('TR')    

    try:

        while run_thread:

             if q_2ansys.empty() is False:

                log.info("\n\nq_2ansys - Get msgFromBonsai")
                msgFromBonsai = q_2ansys.get()
                log.info(f'{"q_2ansys - msgFromBonsai is: {} ".format(msgFromBonsai)}')

                if msgFromBonsai[0] == 'Relaunch':

                    log.info("Relaunch TR")
                    q_2ansys.task_done()
                    tb.analyze_setup('TR')
                             
                    
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




        






