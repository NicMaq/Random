inkling "2.0"

using Number

#Action Space Discreet(3, float32)
#action 0 Cart moves left
#action 1 Cart does not moves
#action 2 Cart moves right
#Observation Shape (3,) Cart Position, Ball y, Ball z

function Reward(gs: BlenderState) {
    return gs._gym_reward
}

function Terminal(gs: BlenderState) {
    return gs._gym_terminal
}

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

    concept Catch(input): SimAction {
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
