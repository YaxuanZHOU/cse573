
MOVE_AHEAD = 'MoveAhead'
ROTATE_LEFT = 'RotateLeft'
ROTATE_RIGHT = 'RotateRight'
LOOK_UP = 'LookUp'
LOOK_DOWN = 'LookDown'
DONE_T = 0  # YZ
DONE_B = 1   # YZ
# DONE_ALL = 2   # YZ

BASIC_ACTIONS = [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, DONE_T, DONE_B]     # YZ

# YZ: VANILLA REWARD
# GOAL_SUCCESS_REWARD = 5
# STEP_PENALTY = -0.01
# FAILED_ACTION_PENALTY = 0

# YZ: REWARD CHANGE 1
# GOAL2_SUCCESS_REWARD = 30   # reward for finding another object
# GOAL_SUCCESS_REWARD = 5     # reward for finding one object
# STEP_PENALTY = -0.01
# FAILED_ACTION_PENALTY = 0

# YZ: REWARD CHANGE 3
GOAL2_SUCCESS_REWARD = 15   # reward for finding another object
GOAL_SUCCESS_REWARD = 5     # reward for finding one object
STEP_PENALTY = -0.01
FAILED_ACTION_PENALTY = -0.05 # penalty for failed action