inkling "2.0"

#using Goal - OPTIONAL
using Goal
using Math
using Number

type SimState {
    Tank_height: number,
    Reference_height: number,
}

type SimAction {
    Set_Pump: number<0.0 .. 1.0 step 0.1>,
}

type SimConfig {
    Reference_height: number<0 .. 3 step 0.5>,
}

#function Terminal(obs: SimState) {
#    return (obs.ItemsInQueue == 100)
#}

#function Reward(obs: SimState) {
#    return (-(obs.ItemsInQueue - 50)**2 + 100) / 2000 
#}

simulator Ansys(action: SimAction, config: SimConfig): SimState {
    #package "AL_Queue_Delay_Azure"
}

graph (input: SimState): SimAction {

    concept DriveHeightToGoal(input): SimAction {
        curriculum {
            training {
                EpisodeIterationLimit : 300,
                NoProgressIterationLimit: 1500000,
            }
            
            source Ansys
            #reward Reward
            #terminal Terminal
            goal (simstate:SimState){
                drive HeightTo: simstate.Tank_height in Goal.Range(simstate.Reference_height-0.5,simstate.Reference_height+0.5)
                avoid HeightMax: simstate.Tank_height in Goal.RangeAbove(3)
            }
            lesson `Constant arrival rate` {
                # These map to the SimConfig values
                scenario {
                    Reference_height: 2,
                }
            }            
        }
    }

}

