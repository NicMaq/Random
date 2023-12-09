#from msilib import sequence
import os
import sys
#import tensorflow as tf
import time
import argparse
import gym
from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface, SimulatorState, SimulatorSessionResponse
from typing import NamedTuple, Dict, Any

workspace = os.getenv("SIM_WORKSPACE")
accesskey = os.getenv("SIM_ACCESS_KEY")

config_client = BonsaiClientConfig()
client = BonsaiClient(config_client)

registration_info = SimulatorInterface(
    name="Mujoco",
    timeout=60,
    simulator_context=config_client.simulator_context,
    description=None,
)

registered_session: SimulatorSessionResponse = client.session.create(workspace_name=config_client.workspace, body=registration_info)
print(f"Registered simulator. {registered_session.session_id}")



class SimulatorModel:
    def __init__(self, args ):

        self.args = args
        if self.args.debug: print('make env')
        
        self.gym_env = gym.make("InvertedDoublePendulum-v4", new_step_api=True, render_mode='human') #'human' 'rgb_array'
        self.init_states = self.gym_env.reset()


    def reset(self, config) -> Dict[str, Any]:

        if self.args.debug: print('reset env with config:', config)
        states = self.gym_env.reset()
        returned_dict = self.gym_to_state(states, 0.0, False)
        
        return { 
            'sim_halted': False,
            'key': returned_dict
        }

    def step(self, action) -> Dict[str, Any]:
        
        if self.args.debug: print('step with action: ', action )
        next_state, step_reward, done1, done2, info = self.gym_env.step(action['action']) # next_state, step_reward, done, info
        returned_dict = self.gym_to_state(next_state, step_reward, done1)
        if self.args.debug: 
            print('next_state is: ', next_state )
            print('step_reward is: ', step_reward )

        return {
            'sim_halted': False,
            'key': returned_dict
        }

    
    def gym_to_state(self, next_state, step_reward, done ):
        state = {
            "pos": float(next_state[0]),
            "sin_hinge1": float(next_state[1]),
            "sin_hinge2": float(next_state[2]),
            "cos_hinge1": float(next_state[3]),
            "cos_hinge2": float(next_state[4]),
            "velocity": float(next_state[5]),
            "ang_velocity1": float(next_state[6]),
            "ang_velocity2": float(next_state[7]),
            "constraint1": float(next_state[8]),
            "constraint2": float(next_state[9]),
            "constraint3": float(next_state[10]),
            "_gym_reward": float(step_reward),
            "_gym_terminal": float(done)    

        }

        return state



def simulate(args, **kwargs):

    sequence_id = 1
    sim_model = SimulatorModel(args)
    sim_model_state = { 'sim_halted': False }

    try:
        while True:
            sim_state = SimulatorState(sequence_id=sequence_id, state=sim_model_state, halted=sim_model_state.get('sim_halted', False))
            if args.debug and sequence_id % 100 == 0: print('sim_state:', sim_state)

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

    
def parse_kw_args(args):
    """
    Parse keyword args into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval    


def main(args):

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--render', help='render the env', action='store_true')                 
    #parser.add_argument(
    #  '--env', help='Select environment')   
    parser.add_argument(
      '--debug', help='create report on model.', action='store_true')         

    kwargs = parse_kw_args(args)
    args = parser.parse_args()  

    #gpus = tf.config.list_physical_devices('GPU')
    #print('GPUS are: ', gpus)

    #for gpu in gpus:
    #    tf.config.experimental.set_memory_growth(gpu,True)

    #wandb.config.update(args)

    #if args.debug:
    #    tf.debugging.set_log_device_placement(True)

    simulator = simulate(args, **kwargs)

if __name__ == '__main__':

    #Init logging
    #wandb.init(project="Eden", entity="nicmaq")
    #wandb.run.log_code(".")
    main(sys.argv)    