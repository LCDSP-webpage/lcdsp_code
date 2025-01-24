# -*- coding:utf-8  -*-
import os
import numpy as np
import torch as th
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))


def concate_observation_from_raw(obs):
    obs_cat = np.hstack([np.array(obs[k], dtype=np.float32).flatten() for k in sorted(obs)])
    obs_cat = th.from_numpy(obs_cat)
    return obs_cat

def concate_observation_from_raw_no_sort(obs):
    obs_cat = np.hstack([np.array(obs[k], dtype=np.float32).flatten() for k in obs])
    obs_cat = th.from_numpy(obs_cat)
    return obs_cat


def concat_state_and_style(states, style_state):
    style_state = np.array(style_state, dtype=np.float32)
    style_state = th.from_numpy(style_state)
    concat_state = th.cat([states, style_state], dim=0)
    return concat_state


class MyFeatureEncoder_nvn:
    def __init__(self, total_player_num):
        self.player_num = total_player_num
        self.active = -1
        self.player_pos_x, self.player_pos_y = 0, 0
        self.last_loffside = np.zeros(self.player_num, np.float32)
        self.last_roffside = np.zeros(self.player_num, np.float32)

    def get_feature_dims(self):
        dims = {
            "player": 19,
            "ball": 18,
            "left_team": 36,
            "left_team_closest": 9,
            "right_team": 45,
            "right_team_closest": 9,
            "avail": 19,
            "match_state": 10,
            "offside": 10,
            "card": 20,
            "sticky_action": 10,
            "ball_distance": 9,
        }
        return dims

    def get_history_feature_dims(self):
        dims = {
            "player": 19,
            "ball": 18,
            "left_team": 32,
            "right_team": 40,
            "offside": 10,
            "ball_distance": 9,
        }
        return dims

    def encode(self, obs):
        player_num = obs["active"]

        player_pos_x, player_pos_y = obs["left_team"][player_num]
        player_direction = np.array(obs["left_team_direction"][player_num])
        player_speed = np.linalg.norm(player_direction)
        player_role = obs["left_team_roles"][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs["left_team_tired_factor"][player_num]
        is_dribbling = obs["sticky_actions"][9]
        is_sprinting = obs["sticky_actions"][8]

        ball_x, ball_y, ball_z = obs["ball"]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        ball_owned = 0.0
        if obs["ball_owned_team"] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs["ball_owned_team"] == 0:
            ball_owned_by_us = 1.0
        elif obs["ball_owned_team"] == 1:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0

        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0

        avail = self._get_avail_new(obs, ball_distance)
        # avail = self._get_avail(obs, ball_distance)
        player_state = np.concatenate(
            (
                # avail[2:],
                obs["left_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                player_role_onehot,
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        player_history_state = np.concatenate(
            (
                obs["left_team"][player_num],
                player_direction * 100,
                [player_speed * 100],
                player_role_onehot,
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )

        ball_state = np.concatenate(
            (
                np.array(obs["ball"]),
                np.array(ball_which_zone),
                np.array([ball_x_relative, ball_y_relative]),
                np.array([obs["ball_direction"][0] * 20, obs["ball_direction"][1] * 20, obs["ball_direction"][2] * 5]),
                np.array(
                    [ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]
                ),
            )
        )

        obs_left_team = np.delete(obs["left_team"], player_num, axis=0)
        obs_left_relatvie = obs_left_team - obs["left_team"][player_num]
        obs_left_team_direction = np.delete(
            obs["left_team_direction"], player_num, axis=0
        )
        left_team_distance = np.linalg.norm(
            obs_left_team - obs["left_team"][player_num], axis=1, keepdims=True
        )
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_tired = np.delete(
            obs["left_team_tired_factor"], player_num, axis=0
        ).reshape(-1, 1)
        left_team_state = np.concatenate(
            (
                obs_left_team,
                obs_left_relatvie * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance,
                left_team_tired,
            ),
            axis=1,
        )
        left_closest_idx = np.argmin(left_team_distance)
        left_closest_state = left_team_state[left_closest_idx]

        left_team_history_state = np.concatenate(
            (
                obs_left_team,
                obs_left_relatvie * 2,
                obs_left_team_direction * 100,
                left_team_speed * 100,
                left_team_distance,
            ),
            axis=1,
        )

        obs_right_team = np.array(obs["right_team"])
        obs_right_relative = obs_right_team - obs["left_team"][player_num]
        obs_right_team_direction = np.array(obs["right_team_direction"])
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["left_team"][player_num], axis=1, keepdims=True
        )
        right_team_speed = np.linalg.norm(
            obs_right_team_direction, axis=1, keepdims=True
        )
        right_team_tired = np.array(obs["right_team_tired_factor"]).reshape(-1, 1)
        right_team_state = np.concatenate(
            (
                obs_right_team,
                obs_right_relative * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance,
                right_team_tired,
            ),
            axis=1,
        )
        right_closest_idx = np.argmin(right_team_distance)
        right_closest_state = right_team_state[right_closest_idx]

        right_team_history_state = np.concatenate(
            (
                obs_right_team,
                obs_right_relative * 2,
                obs_right_team_direction * 100,
                right_team_speed * 100,
                right_team_distance,
            ),
            axis=1,
        )

        steps_left = obs['steps_left']  # steps left till end
        half_steps_left = steps_left
        if half_steps_left > 1500:
            half_steps_left -= 1501  # steps left till halfend
        half_steps_left = 1.0 * min(half_steps_left, 300.0)  # clip
        half_steps_left /= 300.0

        score_ratio = 1.0 * (obs['score'][0] - obs['score'][1])
        score_ratio /= 5.0
        score_ratio = min(score_ratio, 1.0)
        score_ratio = max(-1.0, score_ratio)

        game_mode = np.zeros(7, dtype=np.float32)
        game_mode[obs['game_mode']] = 1
        match_state = np.concatenate(
            (
                np.array([1.0 * steps_left / 3001, half_steps_left, score_ratio]),
                game_mode
            )
        )

        # offside
        l_o, r_o = self.get_offside(obs)
        offside = np.concatenate(
            (
                l_o,
                r_o
            )
        )

        # card
        card = np.concatenate(
            (
                obs['left_team_yellow_card'],
                obs['left_team_active'],
                obs['right_team_yellow_card'],
                obs['right_team_active']
            )
        )

        # sticky_action
        sticky_action = obs['sticky_actions']

        # ball_distance
        left_team_distance = np.linalg.norm(
            obs_left_team - obs["ball"][:2], axis=1, keepdims=False
        )
        right_team_distance = np.linalg.norm(
            obs_right_team - obs["ball"][:2], axis=1, keepdims=False
        )
        ball_distance = np.concatenate(
            (
                left_team_distance,
                right_team_distance
            )
        )
        
        area_vector = self.encode_player_area(player_pos_x, player_pos_y)

        state_dict = {
            "player": player_state,
            "ball": ball_state,
            "left_team": left_team_state,
            "left_closest": left_closest_state,
            "right_team": right_team_state,
            "right_closest": right_closest_state,
            "avail": avail,
            "match_state": match_state,
            "offside": offside,
            "card": card,
            "sticky_action": sticky_action,
            "ball_distance": ball_distance,
            "player_area": area_vector
        }

        history_state_dict = {
            "player": player_history_state,
            "ball": ball_state,
            "left_team": left_team_history_state,
            "right_team": right_team_history_state,
            "offside": offside,
            "ball_distance": ball_distance
        }

        return state_dict, history_state_dict

    def get_offside(self, obs):
        ball = np.array(obs['ball'][:2])
        ally = np.array(obs['left_team'])
        enemy = np.array(obs['right_team'])

        if obs['game_mode'] != 0:
            self.last_loffside = np.zeros(self.player_num, np.float32)
            self.last_roffside = np.zeros(self.player_num, np.float32)
            return np.zeros(self.player_num, np.float32), np.zeros(self.player_num, np.float32)

        need_recalc = False
        effective_ownball_team = -1
        effective_ownball_player = -1

        if obs['ball_owned_team'] > -1:
            effective_ownball_team = obs['ball_owned_team']
            effective_ownball_player = obs['ball_owned_player']
            need_recalc = True
        else:
            ally_dist = np.linalg.norm(ball - ally, axis=-1)
            enemy_dist = np.linalg.norm(ball - enemy, axis=-1)

            if np.min(ally_dist) < np.min(enemy_dist):
                if np.min(ally_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 0
                    effective_ownball_player = np.argmin(ally_dist)

            elif np.min(enemy_dist) < np.min(ally_dist):
                if np.min(enemy_dist) < 0.017:
                    need_recalc = True
                    effective_ownball_team = 1
                    effective_ownball_player = np.argmin(enemy_dist)

        if not need_recalc:
            return self.last_loffside, self.last_roffside

        left_offside = np.zeros(self.player_num, np.float32)
        right_offside = np.zeros(self.player_num, np.float32)

        if effective_ownball_team == 0:
            right_xs = [obs['right_team'][k][0] for k in range(0, self.player_num)]
            right_xs = np.array(right_xs)
            right_xs.sort()

            offside_line = max(right_xs[-2], ball[0])

            for k in range(1, self.player_num):
                if obs['left_team'][k][0] > offside_line and k != effective_ownball_player \
                        and obs['left_team'][k][0] > 0.0:
                    left_offside[k] = 1.0
        else:
            left_xs = [obs['left_team'][k][0] for k in range(0, self.player_num)]
            left_xs = np.array(left_xs)
            left_xs.sort()

            offside_line = min(left_xs[1], ball[0])

            for k in range(1, self.player_num):
                if obs['right_team'][k][0] < offside_line and k != effective_ownball_player \
                        and obs['right_team'][k][0] < 0.0:
                    right_offside[k] = 1.0

        self.last_loffside = left_offside
        self.last_roffside = right_offside

        return left_offside, right_offside

    def _get_avail(self, obs, ball_distance):
        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        (
            NO_OP,
            MOVE,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

        if obs["ball_owned_team"] == 1:  # opponents owning ball
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
        elif (
                obs["ball_owned_team"] == -1
                and ball_distance > 0.03
                and obs["game_mode"] == 0
        ):  # Ground ball  and far from me
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
        else:  # my team owning ball
            avail[SLIDE] = 0

        # Dealing with sticky actions
        sticky_actions = obs["sticky_actions"]
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        # if too far, no shot
        ball_x, ball_y, _ = obs["ball"]
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif (0.64 <= ball_x and ball_x <= 1.0) and (
                -0.27 <= ball_y and ball_y <= 0.27
        ):
            avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

        if obs["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _get_avail_new(self, obs, ball_distance):
        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        (
            NO_OP,
            LEFT,
            TOP_LEFT,
            TOP,
            TOP_RIGHT,
            RIGHT,
            BOTTOM_RIGHT,
            BOTTOM,
            BOTTOM_LEFT,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        if obs["ball_owned_team"] == 1:  # opponents owning ball
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
            if ball_distance > 0.03:
                avail[SLIDE] = 0
        elif (
                obs["ball_owned_team"] == -1
                and ball_distance > 0.03
                and obs["game_mode"] == 0
        ):  # Ground ball  and far from me
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
                avail[SLIDE],
            ) = (0, 0, 0, 0, 0, 0)
        else:  # my team owning ball
            avail[SLIDE] = 0
            if ball_distance > 0.03:
                (
                    avail[LONG_PASS],
                    avail[HIGH_PASS],
                    avail[SHORT_PASS],
                    avail[SHOT],
                    avail[DRIBBLE],
                ) = (0, 0, 0, 0, 0)

        # Dealing with sticky actions
        sticky_actions = obs["sticky_actions"]
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        # if too far, no shot
        ball_x, ball_y, _ = obs["ball"]
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif (0.64 <= ball_x <= 1.0) and (
                -0.27 <= ball_y <= 0.27
        ):
            avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

        if obs["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
                -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (
                -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
                -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 0, 0, 1.0, 0]
        else:
            return [0, 0, 0, 0, 0, 1.0]

    def _encode_role_onehot(self, role_num):
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result[role_num] = 1.0
        return np.array(result)
    
    def encode_player_area(self, x, y):
        MIDDLE_X, END_X = 0.3, 1.0
        MIDDLE_Y, END_Y = 0.21, 0.42
        
        x_area = [0, 0, 0]
        y_area = [0, 0, 0]
        
        if MIDDLE_X < x <= END_X:
            x_area = [1, 0, 0]
        elif -MIDDLE_X <= x <= MIDDLE_X:
            x_area = [0, 1, 0]
        elif -END_X <= x < -MIDDLE_X:
            x_area = [0, 0, 1]
        
        if -END_Y <= y < -MIDDLE_Y:
            y_area = [1, 0, 0]
        elif -MIDDLE_Y <= y <= MIDDLE_Y:
            y_area = [0, 1, 0]
        elif MIDDLE_Y < y <= END_Y:
            y_area = [0, 0, 1]
        
        return x_area + y_area  

class SamplerAgent(object):
    def __init__(
            self,
            sampler,
            obs_type,
            model_type=-1
    ):
        self.sampler = sampler
        self.obs_type = obs_type
        self.model_type = model_type
        self.pnet = None

    def process_reward(self, reward):
        return reward

    def process_action_mask(self, obs, use_mask=False):
        s = '''
        owner_ship = obs[142] > 0.5
        movement = sum(obs[-21:-13]) > 0.5
        sprint = obs[-13] > 0.5
        dribble = obs[-12] > 0.5
        if movement:
            move_release = 1
        else:
            move_release = 0
        if sprint:
            sprint_release = 1
        else:
            sprint_release = 0
        if dribble:
            dribble_release = 1
        else:
            dribble_release = 0
        if owner_ship:
            p_s = [1, 1, 1, 1]
        else:
            p_s = [0, 0, 0, 0]
        mask = [1 for _ in range(9)] + p_s + [1, move_release, sprint_release,
                                              1, 1, dribble_release]

        mask = th.from_numpy(np.asarray(mask, dtype=np.float32))
        mask = (1 - mask).to(th.bool)
'''
        if not use_mask:
            mask = [1 for _ in range(19)]
            mask = th.from_numpy(np.asarray(mask, dtype=np.float32))
            mask = (1 - mask).to(th.bool)
            return mask

        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        (
            NO_OP,
            LEFT,
            TOP_LEFT,
            TOP,
            TOP_RIGHT,
            RIGHT,
            BOTTOM_RIGHT,
            BOTTOM,
            BOTTOM_LEFT,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        player_num = obs["active"]
        player_pos_x, player_pos_y = obs["left_team"][player_num]
        ball_x, ball_y, _ = obs["ball"]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])

        if (obs["ball_owned_team"] == 1
                and ball_distance > 0.05):  # opponents owning ball
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
            if ball_distance > 0.03:
                avail[SLIDE] = 0
        elif (
                obs["ball_owned_team"] == -1
                and ball_distance > 0.05
                and obs["game_mode"] == 0
        ):  # Ground ball  and far from me
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
                avail[SLIDE],
            ) = (0, 0, 0, 0, 0, 0)
            if ball_distance > 0.03:
                avail[SLIDE] = 0
        else:  # my team owning ball
            avail[SLIDE] = 0
            if ball_distance > 0.05:
                (
                    avail[LONG_PASS],
                    avail[HIGH_PASS],
                    avail[SHORT_PASS],
                    avail[SHOT],
                    avail[DRIBBLE],
                ) = (0, 0, 0, 0, 0)

        # Dealing with sticky actions
        sticky_actions = obs["sticky_actions"]
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        mask = th.from_numpy(np.asarray(avail, dtype=np.float32))
        mask = (1 - mask).to(th.bool)

        return mask

    def process_image(self, image):
        # image is w x h x c
        image = image.transpose(2, 1, 0)
        return th.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

    def fetch_model_parameters(self, models):
        self.pnet = th.jit.load(models, map_location=th.device('cpu'))

    def get_model_result(self, states, amask):
        with th.no_grad():
            if len(states.shape) == 1:
                states = th.unsqueeze(states, 0)
            if len(amask.shape) == 1:
                amask = th.unsqueeze(amask, 0)
            prob, log_prob, state_value = self.pnet(states, amask)
            action = prob.multinomial(1).item()
            # action = prob.argmax(1).item()
            return action, prob, log_prob, state_value

    def get_model_batch_result(self, states, amask):
        with th.no_grad():
            if isinstance(states, list):
                states = th.stack(states, 0)
                amask = th.stack(amask, 0)

            if len(states.shape) == 1:
                states = th.unsqueeze(states, 0)
            if len(amask.shape) == 1:
                amask = th.unsqueeze(amask, 0)
            prob, log_prob, state_value = self.pnet(states, amask)
            action = prob.multinomial(1)
            return action, prob, log_prob, state_value

    def get_model_argmax_result(self, states, amask):
        with th.no_grad():
            if isinstance(states, list):
                states = th.stack(states, 0)
                amask = th.stack(amask, 0)

            if len(states.shape) == 1:
                states = th.unsqueeze(states, 0)
            if len(amask.shape) == 1:
                amask = th.unsqueeze(amask, 0)
            prob, log_prob, state_value = self.pnet(states, amask)
            action = prob.argmax(1)
            return action, prob, log_prob, state_value


feature_encoder = MyFeatureEncoder_nvn(2)
models = os.path.dirname(os.path.abspath(__file__)) + '/model.pth'
agent = SamplerAgent(sampler=None, obs_type=1)
agent.fetch_model_parameters(models)


def my_controller(observation, action_space, is_act_continuous=False, style=None):
    states, history_state = feature_encoder.encode(observation)
    states = concate_observation_from_raw_no_sort(states)
    if style:
        states = concat_state_and_style(states, style)
    amask = agent.process_action_mask(observation, use_mask=True)
    action, prob, log_prob, state_value = agent.get_model_result(states, amask)

    action_final = [[0] * 19]
    action_final[0][action] = 1
    return action_final


def my_controller_batch(batch_obs, action_space, is_act_continuous=False, style=None):
    action_final = []
    states_list, mask_list = [], []
    for obs in batch_obs:
        states, history_state = feature_encoder.encode(obs)
        states = concate_observation_from_raw_no_sort(states)
        if style:
            states = concat_state_and_style(states, style)
        amask = agent.process_action_mask(obs, use_mask=True)
        states_list.append(states)
        mask_list.append(amask)
    action, prob, log_prob, state_value = agent.get_model_batch_result(states_list, mask_list)
    if len(action.shape) == 2:
        action = action.squeeze(1).numpy().tolist()
    else:
        action = action.numpy().tolist()
    for act in action:
        action_one_hot = [[0] * 19]
        action_one_hot[0][act] = 1
        action_final.append(action_one_hot)
    return action_final


def my_controller_argmax(batch_obs, action_space, is_act_continuous=False, style=None):
    action_final = []
    states_list, mask_list = [], []
    for obs in batch_obs:
        states = feature_encoder.encode(obs)
        states = concate_observation_from_raw_no_sort(states)
        if style:
            states = concat_state_and_style(states, style)
        amask = agent.process_action_mask(obs, use_mask=False)
        states_list.append(states)
        mask_list.append(amask)
    action, prob, log_prob, state_value = agent.get_model_argmax_result(states_list, mask_list)
    if len(action.shape) == 2:
        action = action.squeeze(1).numpy().tolist()
    else:
        action = action.numpy().tolist()
    for act in action:
        action_one_hot = [[0] * 19]
        action_one_hot[0][act] = 1
        action_final.append(action_one_hot)
    return action_final