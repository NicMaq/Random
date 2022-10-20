%%isaac
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
