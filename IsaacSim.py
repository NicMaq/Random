from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.stage import create_new_stage_async
from omni.kit.viewport_legacy import get_default_viewport_window
from abc import abstractmethod
import asyncio
import gc
import carb

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.tasks import BaseTask

from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.wheeled_robots.robots import WheeledRobot
import numpy as np

from solutions.JetbotControllers import MoveToPointController

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.tasks import BaseTask
import numpy as np

## Old Imports
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.wheeled_robots.robots import WheeledRobot

## New Imports
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import PickPlaceController

class SimpleBase3:
    
    '''Simple Interface with Kit for some simple extensions'''
    
    def set_initial_camera_params(self, cam_center=[2,2,1], cam_target=[0,0,0]):
        ## Sets camera starting position and viewing target
        cam_name = "/OmniverseKit_Persp"
        viewport = get_default_viewport_window()
        viewport.set_camera_position(cam_name, *cam_center[:3], True)
        viewport.set_camera_target  (cam_name, *cam_target[:3], True)
    
    async def load_world_async(self, start, new_stage, **kwargs):
        ## Get or create a new clean world
        if World.instance() is None or new_stage:
            self._world = World(
                stage_units_in_meters = 1.,
                physics_dt   = 1./60, 
                rendering_dt = 1./60)
            await create_new_stage_async()
        else:
            self._world = World.instance()
            self._world.stop()
            self._world.clear()
            gc.collect()

        ## Initialize context, scene setup, and world reset/pause
        await self._world.initialize_simulation_context_async()
        self.set_initial_camera_params(**kwargs)
        if hasattr(self, "set_up_scene"):
            self.set_up_scene(self._world.scene)
        
        await self._world.reset_async() 
        await self._world.pause_async() 
        
        if hasattr(self, "post_reset"):
            self.post_reset()
        
        ## Make the extension-wide pre_step method optional
        if hasattr(self, "pre_step"):
            self._world.add_physics_callback("pre_step", self.pre_step)
        
        ## If there are any tasks, make sure they do per-step actions
        if len(self._world.get_current_tasks()) > 0:
            self._world.add_physics_callback("tasks_step", self._world.step_async)

        if start: await self._world.play_async() 
            
    def load(self, start=True, new_stage=False, **kwargs):
        asyncio.get_running_loop().create_task(
            self.load_world_async(start, new_stage, **kwargs))

class JetbotManager(BaseTask):
    
    def __init__(
        self, 
        name, 
        start_position = np.array([.0, .0, .0]), 
    ):
        super().__init__(name)
        self._start_pos = start_position
        self._goals = []
        self._state = 0
    
    def set_up_scene(self, scene):
        self._scene = scene ## This will be useful later
        self._jetbot = scene.add(WheeledRobot(
            prim_path=f"/World/{self.name}/jetbot",
            name=f"{self.name}_jetbot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            position=self._start_pos, 
            usd_path=f"{get_assets_root_path()}/Isaac/Robots/Jetbot/jetbot.usd",
        ))
    
    def post_reset(self):
        self._controller = MoveToPointController(name="robot_control")
        
    def move_to_pos(self, goal_pos):
        self._goals += [goal_pos]
    
    def pre_step(self, control_index, simulation_time):

        if self._state >= len(self._goals): return
        
        curr_goal = self._goals[self._state]
        position, orientation = self._jetbot.get_world_pose()
        action = self._controller.forward(
            start_position    = position,
            start_orientation = orientation,
            goal_position     = curr_goal
        )
        self._jetbot.apply_action(action)
        
        if np.linalg.norm(curr_goal - position) < 0.05:
            self._increase_state()

    def _increase_state(self):   ## This will be useful later
        self._state += 1
            

class FrankaManager(BaseTask):
        
    def __init__(
        self, 
        name, 
        start_position = np.array([.0, .0, .0]), 
    ):
        super().__init__(name)
        ## Start position of Franka
        self._start_pos = start_position
        self._goals = [] 
        self._state = 0
        ## See new methods to see what these do...
        self._cube_name = None 
        self._cube_names = []  
        self._cubes = []
    
    def set_up_scene(self, scene):
        self._scene = scene
        self._franka = scene.add(Franka(
            prim_path=f"/World/{self.name}/franka",
            name=f"{self.name}_franka",
            position = self._start_pos
        ))
    
    def post_reset(self):
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper_dof_indices = self._franka.gripper.dof_indices,
            robot_articulation  = self._franka
        )
        self._cubes = [self._scene.get_object(name) for name in self._cube_names]
    
    def pre_step(self, control_index, simulation_time):
        if self._state >= len(self._goals): return
        
        curr_goal = self._goals[self._state]
        curr_cube = self._cubes[self._state]
        action = self._controller.forward(
            picking_position = curr_cube.get_world_pose()[0],
            placing_position = curr_goal,
            current_joint_positions = self._franka.get_joint_positions()
        )
        self._franka.apply_action(action)
        
        if self._controller.is_done():
            self._controller.reset()
            self._increase_state()

    def _increase_state(self):
        self._state += 1
    
    '''BEGIN NEW METHODS'''
    def set_cube_name(self, cube_name):
        ## current cube name for which goals will be added
        self._cube_name = cube_name
        
    def move_cube_to_pos(self, goal_pos):
        ## Cube named _cube_names[i] which will be moved to _goals[i]        
        self._cube_names += [self._cube_name]
        self._goals += [goal_pos]
    '''END NEW METHODS'''

class FrankaManager2(FrankaManager):
    
    def __init__(self, name, start_position = np.array([.0, .0, .0])): 
        super().__init__(name, start_position)
        self._task_states = []
        
    def move_cube_to_pos(self, goal_pos, task_state):
        super().move_cube_to_pos(goal_pos)
        self._task_states += [task_state]
        
    def _increase_state(self):
        self._state += 1
        self._scene.task_state += 1

    def pre_step(self, control_index, simulation_time):
        if self._state >= len(self._task_states): return
        if self._scene.task_state != self._task_states[self._state]: return
        super().pre_step(control_index, simulation_time)
            
class JetbotManager2(JetbotManager):
        
    def __init__(self, name, start_position = np.array([.0, .0, .0])): 
        super().__init__(name, start_position)
        self._task_states = []
        
    def move_to_pos(self, goal_pos, task_state):
        super().move_to_pos(goal_pos)
        self._task_states += [task_state]
        
    def _increase_state(self):
        self._state += 1
        self._scene.task_state += 1

    def pre_step(self, control_index, simulation_time):
        if self._state >= len(self._task_states): return
        if self._scene.task_state != self._task_states[self._state]: return
        super().pre_step(control_index, simulation_time)
        
        class ExerciseFrankaAndJetbot(SimpleBase3):
        
    def set_up_scene(self, scene):
        scene.add_default_ground_plane()
        scene.task_state = 0
        
        '''TODO: Figure out where you want to drop your initial cube'''
        cube_pos = np.array([1, -1, 1])

        '''TODO: Figure out where you want your Franka to move your cube'''
        franka_goal = np.array([ .3,  .3,  .5])
        
        cube = scene.add(DynamicCuboid(
            prim_path = f"/World/cube", 
            name      = f"cube",
            position  = cube_pos,
            size      = np.array([.05, .05, .05]), 
            color     = np.array([.0, 1., .0]),  
        ))
        franka = FrankaManager2('franka_task')
        jetbot = JetbotManager2('jetbot_task', start_position=np.array([1, 1, 0]))
        
        franka.set_cube_name(cube.name)
        '''TODO: Move the Jetbot to a position where it can get the cube
        You can do this manually or set up a heuristic which will always 
        position it to where it can scoop up nicely.
        
        HINT: Just give move_to_pos a 0 as the task_state. 
        Increase this for subsequent commands
        '''
        jetbot.move_to_pos(cube_pos * 1.5, 0)
        
        '''TODO: Tell the jetbot to pull in the cube until it is in 
        reach of the franka.
        '''
        jetbot.move_to_pos(cube_pos / 3, 1)
        '''TODO: Move the jetbot away so it doesn't get struck by the franka
        '''
        jetbot.move_to_pos(cube_pos / 2 , 2)
        '''TODO: Tell the franka to pick up the cube and move it somewhere
        '''
        franka.move_cube_to_pos(franka_goal, 3)
        '''TODO: Touch the cube with the Jetbot
        '''
        jetbot.move_to_pos(franka_goal, 4)
        self._world.add_task(franka)
        self._world.add_task(jetbot)
            
ExerciseFrankaAndJetbot().load()
