# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from . import *
import random


player_role_list = [e_PlayerRole_RM, e_PlayerRole_CF, e_PlayerRole_LB, e_PlayerRole_CB]
random.shuffle(player_role_list)
print(player_role_list)

episode = 0
stored_ball_position_x = None
stored_ball_position_y = None

from . import *

def build_scenario(builder):
    global episode, stored_ball_position_x, stored_ball_position_y
    episode += 1
    # print("enter_dynamic", episode)
    
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = False
    builder.config().end_episode_on_possession_change = False
    
    if episode % 5 == 1:
        stored_ball_position_x = random.uniform(-0.99, 0.7)
        stored_ball_position_y = random.uniform(-0.41, 0.41)
    # Use stored positions
    ball_position_x = stored_ball_position_x
    ball_position_y = stored_ball_position_y

    builder.SetBallPosition(ball_position_x, ball_position_y)
     
    builder.SetBallPosition(ball_position_x, ball_position_y)    

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, True)
    builder.AddPlayer(ball_position_x + random.uniform(-0.01, 0.01), ball_position_y + random.uniform(-0.01, 0.01), random.choice(player_role_list))
    builder.AddPlayer(random.uniform(-0.99, 0.75), random.uniform(-0.41, 0.41), random.choice(player_role_list))

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, True)
    builder.AddPlayer(random.uniform(-0.99, -0.8), random.uniform(-0.41, 0.41), random.choice(player_role_list))
    builder.AddPlayer(random.uniform(-0.99, -0.9), random.uniform(-0.41, 0.41), random.choice(player_role_list))
