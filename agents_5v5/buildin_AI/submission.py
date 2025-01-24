import enum
from functools import wraps
from typing import *
import math
import random

def my_controller(observation, action_space, is_act_continuous=False):
    action_ = [[0] * 20]
    action_[0][19] = 1
    return action_

def my_controller_batch(batch_obs, action_space, is_act_continuous=False):
    action_ = [[[0] * 20] for _ in range(len(batch_obs))]
    for action in action_:
        action[0][19] = 1
    return action_