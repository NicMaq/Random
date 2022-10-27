inkling "2.0"

#using Goal - OPTIONAL
using Goal
using Math
using Number

#Action Space Box(-1.0, 1.0, (1,), float32)
#action 0 Force applied on the cart -1 1 slider slide Force (N)
#Observation Shape (11,)
#Observation High [inf inf inf inf inf inf inf inf inf inf inf]
#Observation Low [-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf]

#0 - position of the cart along the linear surface - position (m)
#1 - sine of the angle between the cart and the first pole - sin(hinge)
#2 - sine of the angle between the two poles - sin(hinge2)
#3 - cosine of the angle between the cart and the first pole - cos(hinge)
#4 - cosine of the angle between the two poles - cos(hinge2)
#5 - velocity of the cart - velocity (m/s)
#6 - angular velocity of the angle between the cart and the first pole - angular velocity (rad/s)
#7 - angular velocity of the angle between the two poles - angular velocity (rad/s)
#8 - constraint force - 1 - Force (N) - see https://mujoco.readthedocs.io/en/latest/computation.html
#9 - constraint force - 2 - Force (N)
#10 - constraint force - 3 - Force (N)

#Import gym.make("InvertedDoublePendulum-v4")

function Reward(gs: BlenderState) {
    return gs._gym_reward
}

function Terminal(gs: BlenderState) {
    return gs._gym_terminal
}

const max_position = 100 #m
const max_speed = 100 #m/s
const max_ang_speed = 100 #rad/s
const max_constraint = 100 #N


#"state": {
#    "cart_pos": 0,
#    "ball_y": 3,
#    "ball_z": 16,
#    "_gym_reward": 0,
#    "_gym_terminal": 0
#},

type BlenderState {
    cart_pos: Number.Float32,
    ball_y: Number.Float32,
    ball_z: Number.Float32,
    _gym_reward: number,
    _gym_terminal: number
}

type ObservableState {
    posCart: Number.Float32,
    posBall_y: Number.Float32,
    posBall_z: Number.Float32,
}

type BlenderAction {
    action: Number.Int8<0 .. 2>,
}

type SimAction {
    action: Number.Int8<0 .. 2>,
}

type SimConfig {
    nballs: Number.UInt8
}

function TransformState(state: BlenderState): ObservableState {

    return {
        posCart: state.cart_pos,
        posBall_y: state.ball_y,
        posBall_z: state.ball_z,
    }
}

function TransformAction(state: SimAction): BlenderAction {

    var command: Number.Int8<0 .. 2> = state.action

    return {
        action: command
    }
}

simulator Blender(action: BlenderAction, config: SimConfig): BlenderState {
}

graph (input: ObservableState): SimAction {

    concept StayUp(input): SimAction {
        curriculum {
            source Blender
            reward Reward
            terminal Terminal
            state TransformState
            action TransformAction
            lesson first_lesson {
                scenario {
                    nballs: 1
                }
            }
        }
    }
}
