import json
import numpy as np
import random
import math
import os
import time
from copy import deepcopy
# from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Dependency-free quaternion helpers
# ---------------------------------------------------------------------------
# NOTE: This file historically uses two quaternion conventions:
#   - Omnigibson APIs generally use (x, y, z, w)
#   - Some helper math in this file uses (w, x, y, z)
# We keep the original conventions and behavior intact, but remove external
# deps (pybullet / transforms3d) by providing small equivalents.

def _euler_xyz_to_quat_wxyz(roll, pitch, yaw):
    """Euler (roll, pitch, yaw) -> quaternion (w, x, y, z)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=float)

def _quat_wxyz_to_euler_xyz(q):
    """Quaternion (w, x, y, z) -> Euler (roll, pitch, yaw)."""
    q = np.array(q, dtype=float).reshape(4,)
    w, x, y, z = q

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # numerical safety
    if sinp >= 1.0:
        pitch = math.pi / 2.0
    elif sinp <= -1.0:
        pitch = -math.pi / 2.0
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=float)

def qmult(q1, q2):
    """Quaternion multiplication in (w, x, y, z) order."""
    a = np.array(q1, dtype=float).reshape(4,)
    b = np.array(q2, dtype=float).reshape(4,)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=float)

def euler2quat(ai, aj, ak):
    # Match transforms3d.euler.euler2quat default axes='sxyz'
    return _euler_xyz_to_quat_wxyz(ai, aj, ak)

def quat2euler(q):
    # Match transforms3d.euler.quat2euler default axes='sxyz'
    return _quat_wxyz_to_euler_xyz(q)

class _PyBulletCompat:
    """Tiny subset of pybullet API used by this file."""
    @staticmethod
    def getQuaternionFromEuler(e):
        # pybullet returns (x, y, z, w)
        roll, pitch, yaw = float(e[0]), float(e[1]), float(e[2])
        wxyz = _euler_xyz_to_quat_wxyz(roll, pitch, yaw)
        w, x, y, z = wxyz
        return (x, y, z, w)

    @staticmethod
    def getEulerFromQuaternion(q_xyzw):
        # input is (x, y, z, w)
        x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
        return _quat_wxyz_to_euler_xyz((w, x, y, z))

# Keep the original variable name used throughout the file
p = _PyBulletCompat()

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import REGISTERED_OBJECTS
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.tasks import REGISTERED_TASKS
from omnigibson.scenes import REGISTERED_SCENES
from omnigibson.utils.gym_utils import GymObservable, recursively_generate_flat_dict
from omnigibson.utils.config_utils import parse_config
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.python_utils import assert_valid_key, merge_nested_dicts, create_class_from_registry_and_config,\
    Recreatable
from omnigibson.envs.env_base import Environment
from omnigibson import object_states

# Create module logger
log = create_module_logger(module_name=__name__)


class UcON_Environment(Environment):

    def __init__(self,
                 configs,
                 id=None,
                 action_timestep=1 / 60.0,
                 physics_timestep=1 / 60.0,
                 device=None,
                 automatic_reset=False,
                 flatten_action_space=False,
                 flatten_obs_space=False,
                 block_size=3.0,
                 max_threshold=2.0,
                 min_threshold=0.5,
                 open_dist=0.7,
                 args=None):

        self.args = args
        self.init_angle = np.deg2rad(np.arange(0, 360, 90))
        self.init_quaternion = [p.getQuaternionFromEuler([angle, 0, 0]) for angle in self.init_angle]
        self.init_quaternion = [(x, y, z, w) for w, x, y, z in self.init_quaternion]
        self.step_size = args.step_size
        self.turn_size = args.turn_size
        self._task_path = args.task_path
        self._configs = configs
        self.choose_num = args.target_obj_num
        self.id = id
        self.change_task_cnt = 0
        self.max_step = args.max_step if 'max_step' in args else 50
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.open_dist = open_dist
        self.block_size = block_size
        self.look_angle = 0
        self.init_dist = 0
        self.leave_dist = 2.0
        self.find_dist = 1.0
        self.prev_dist = -1
        self.exploration_map = set()
        self.step_cnt = 0
        self.orientation_part_list = [-1, -np.sqrt(2) / 2, -1 / 2, 0, np.sqrt(2) / 2, 1 / 2, 1]
        self.init_cam_height = 1

        # for test
        self.place_cnt = 0
        self.success_place = 0

        with open(self._task_path, 'r') as f:
            task_data = json.load(f)
        scene_list = list(task_data.keys())
        scene = random.choice(scene_list)
        if args.scene != 'default':
            scene = args.scene
        self.scene_name = scene
        configs['scene']['scene_model'] = scene
        
        while True:
            log.info('Start loading scene: {}'.format(scene))
            self.restore_config_json(scene)

            # the task used for initializing the env
            self._task_list = sample_unique_dicts(task_data[scene], self.choose_num)
            self._obj_location = [random.choice(task_dict['location']) for task_dict in self._task_list]
            self.place_obj = {}
            self.place_cnt += len(self._task_list)

            # modify the best_json for the task
            ref_obj = get_ref_obj(self._obj_location)
            self._ref_obj = deepcopy(ref_obj)
            self.add_obj = []
            self.improve_config_json(scene, ref_obj)

            try:
                og.sim.stop()
            except:
                pass
            super().__init__(configs=configs, action_timestep=action_timestep, physics_timestep=physics_timestep, device=device, automatic_reset=automatic_reset, flatten_action_space=flatten_action_space, flatten_obs_space=flatten_obs_space, id=self.id)
            og.sim.play()

            # self.init_robot_quaternion = self.robots[0].get_orientation()

            # place the target obj
            self._target_obj = []
            self.init_place()
            self._target_obj_num = len(self._target_obj)
            self._task_list = [task for task in self._task_list if task['basic_object_name'] in \
                [obj.category for obj in self._target_obj]]
            self.habit_knowledge = []
            for task in self._task_list:
                self.habit_knowledge.extend(task['habits'])
            random.shuffle(self.habit_knowledge)
            if len(self._task_list) > 0 or self.args.map_collect > 0:
                break
            else:
                pass

        self.state = og.sim.scene.dump_state(serialized=False)
        map_xy_list = list(self.scene.trav_map.floor_graph[0].nodes())
        self.world_xy_list = [self.scene.trav_map.map_to_world(np.array(xy)) for xy in map_xy_list]

        self.now_task = random.choice(self._task_list)
        self.now_target_name = self.now_task['location'][0].split('.')[0]
        for target in self._target_obj:
            if target.category == self.now_target_name:
                self.now_target = target
                break

        self._task_list.remove(self.now_task)

        self.agent = og.sim.viewer_camera
        ori = self.agent.get_orientation()
        for idx in range(len(ori)):
            ori[idx] = min(self.orientation_part_list, key=lambda x: abs(ori[idx] - x))
        self.agent.set_orientation(ori)
        self.agent_start_ori = self.agent.get_orientation()
        # self.place_agent()
        self.init_dist = random.choice([self.leave_dist, self.find_dist])
        self.place_agent_near_obj(self.now_target, dist=self.init_dist)
        self.agent_start_pos = self.agent.get_position()
        self.agent_start_ori = self.agent.get_orientation()
        self.cam_height = self.init_cam_height

        # action space
        self.action_list = ['no_action', 'done', 'move_forward', 'move_backward'
                            'turn_left', 'turn_right', 'look_up', 'look_down', 'open', 'leave']
        self.action_function = {
            'done': self.done,
            'move_forward': self.move_forward,
            'move_backward': self.move_backward,
            'turn_left': self.turn_left,
            'turn_right': self.turn_right,
            'look_up': self.look_up,
            'look_down': self.look_down,
            'open': self.open,
            'leave': self.leave,
            'test': self.test,
            'no_action': self.no_action
        }

    def reset(self, change_scene=False):
        """
        Reset episode.
        """
        og.sim.load_state(self.state, serialized=False)
        self.prev_dist = -1
        self.exploration_map = set()

        with open(self._task_path, 'r') as f:
            task_data = json.load(f)

        self.max_step = 0
        self.change_task_cnt += 1
        if self.change_task_cnt > 1:
            self.step_cnt = 0
            self.change_task_cnt = 0
            no_change_scene = bool(len(self._task_list))
            if not no_change_scene:
                scene_list = list(task_data.keys())
                scene = self.scene_name
                self._configs['scene']['scene_model'] = self.scene_name

                while True:
                    self.restore_config_json(scene)
                    self._task_list = sample_unique_dicts(task_data[scene], self.choose_num)
                    self._obj_location = [random.choice(task_dict['location']) for task_dict in self._task_list]
                    self.place_cnt += len(self._task_list)

                    # modify the best_json for the task
                    ref_obj = get_ref_obj(self._obj_location)
                    self._ref_obj = deepcopy(ref_obj)
                    self.add_obj = []
                    self.improve_config_json(scene, ref_obj)

                    og.sim.stop()
                    self.reload(configs=self._configs)
                    og.sim.viewer_camera.SEMANTIC_REMAPPER.clear()
                    og.sim.play()

                    # place the target obj
                    self._target_obj = []
                    self.init_place()
                    self._target_obj_num = len(self._target_obj)
                    self._task_list = [task for task in self._task_list if task['basic_object_name'] \
                         in [obj.category for obj in self._target_obj]]
                    self.habit_knowledge = []
                    for task in self._task_list:
                        self.habit_knowledge.extend(task['habits'])
                    random.shuffle(self.habit_knowledge)
                    if len(self._task_list) > 0 or self.args.map_collect > 0:
                        break
                    else:
                        pass

                print(self.success_place / self.place_cnt)

                map_xy_list = list(self.scene.trav_map.floor_graph[0].nodes())
                self.world_xy_list = [self.scene.trav_map.map_to_world(np.array(xy)) for xy in map_xy_list]

                # set sleep to disable physics
                for obj in self.scene.objects:
                    if og.object_states.open_state.Open not in obj.states.keys():
                        obj.sleep()

            self.now_task = random.choice(self._task_list)
            self.now_target_name = self.now_task['location'][0].split('.')[0]
            for target in self._target_obj:
                if target.category == self.now_target_name:
                    self.now_target = target
                    break
            self._task_list.remove(self.now_task)

        for obj in self.scene.objects:
            try:
                obj.get_position()
            except:
                og.sim.stop()
                og.sim.play()
                print('refresh')
            if np.any(np.isnan(obj.get_position())):
                og.sim.stop()
                og.sim.play()
                print('refresh')
        self.state = og.sim.scene.dump_state(serialized=False)

        # keep the agent new
        self.agent = og.sim.viewer_camera
        for i in range(random.choice([0, 1, 2, 3])):
            self.turn_left()

        # keep the agent new
        self.agent = og.sim.viewer_camera
        ori = self.agent.get_orientation()
        for idx in range(len(ori)):
            ori[idx] = min(self.orientation_part_list, key=lambda x: abs(ori[idx] - x))
        self.agent.set_orientation(ori)
        self.agent_start_ori = self.agent.get_orientation()
        # self.place_agent()
        # self.init_dist = random.choice([self.leave_dist, self.find_dist])
        # self.place_agent_near_obj(self.now_target, dist=self.init_dist)
        if self.args.map_collect > 0:
            self.place_agent_zero()
        else:
            self.place_agent()

        self.agent_start_pos = self.agent.get_position()
        self.agent_start_ori = self.agent.get_orientation()
        self.exploration_map = set()
        self.look_angle = 0
        self.success_reward_sum, self.dist_reward_sum, self.exploration_reward_sum, self.block_penalty_sum = 0, 0, 0, 0

        # Reset internal variables
        super()._reset_variables()

        # Run a single simulator step to make sure we can grab updated observations
        og.sim.my_step()
        og.sim.my_step()

        # Grab observations
        obs, info = self.agent.get_obs()
        rgb = obs['rgb']
        rgb = rgb[:, :, :3]
        depth = obs['depth_linear']
        # depth = 1 / (depth + 1e-10)
        # depth = depth / 100
        depth[depth > 5] = 5
        bbox_lst = get_bbox(obs, info)
        visible = [bbox['semanticLabel'] for bbox in bbox_lst]
        visible = set(visible)
        # depth_linear = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        agent_pos = self.agent.get_position()
        agent_ori = self.agent.get_orientation()
        agent_ori = agent_ori[[3, 0, 1, 2]]
        # agent_ori = quaternion.quaternion(*agent_ori)
        self.current_episode_init_position = agent_pos
        self.current_episode_init_orientation = quat2euler(agent_ori)
        gps_compass = np.zeros(3)
        # obs = {'rgb': rgb, 'depth': depth, 'pos': agent_pos, 'ori': agent_ori, 'instruction': self.now_task['user_instruction'], 'prompt': self.now_task['prompt'], 'gps_compass': gps_compass,'task':self.now_task, 'visible':visible, 'location': self.now_location }
        obs = {
            'rgb': rgb,
            'depth': depth,
            'seg_semantic': obs['seg_semantic'],
            'info': info,
            'pos': agent_pos,
            'ori': agent_ori,
            'instruction': self.now_task['user_instruction'],
            'habit_knowledge': self.habit_knowledge,
            'prompt': self.now_task['prompt'],
            'gps_compass': gps_compass,
            'task': self.now_task,
            'visible': visible,
            'bbox': bbox_lst,
            'collision': False,
            'turn_collision': False
        }

        return obs

    def place_agent_zero(self):
        land_success = self.agent.set_pos_ori_with_no_collision((0, 0, self.init_cam_height), quat=self.init_quaternion[0])
        return land_success

    def load(self):
        """
        Load the scene and robot specified in the config file.
        """
        # This environment is not loaded
        self._loaded = False

        # Load config variables
        self._load_variables()

        # Load the scene, robots, and task
        self._load_scene()
        # self._load_robots()
        self._load_objects()
        self._load_task()

        og.sim.play()
        self.task.reset(self)
        self._reset_variables()
        og.sim.my_step()
        og.sim.my_step()

        # Load the obs / action spaces
        self.load_observation_space()
        self._load_action_space()

        # Denote that the scene is loaded
        self._loaded = True

    def close(self):
        """
        restore the best_json
        """
        super().close()

        self.restore_config_json(self.scene_name)

    def init_place(self):
        """
        set the initial place of target objects generated by gpt4
        """
        if self.args.map_collect > 0:
            return

        self.place_obj = {}
        self._target_obj = []
        all_ref, ref_obj_must_use = set(), set()

        all_num, cnt = 0, 0
        for location in self._obj_location:

            for obj in self.scene.objects:
                try:
                    obj.get_position()
                except:
                    og.sim.stop()
                    og.sim.play()
                    print('refresh')
                if np.any(np.isnan(obj.get_position())):
                    og.sim.stop()
                    og.sim.play()
                    print('refresh')

            target_obj_name, rest = location.split('.')
            place_action, ref_obj_name = rest.split('(')
            ref_obj_name = ref_obj_name.replace(')', '')
            # choose the object to use
            target_obj, ref_obj = None, None

            # if target_obj_name.split('_')[-1].isdigit():
            #     target_obj = self.scene.object_registry._objects_by_name[target_obj_name]
            # else:
            to_remove = []
            try:
                target_objs = self.scene.object_registry._objects_by_category[target_obj_name]
                target_obj = random.choice(list(target_objs))
                for obj in target_objs:
                    if obj != target_obj:
                        to_remove.append(obj)
                for obj in to_remove:
                    # self.scene.remove_object(obj)
                    obj.set_position([0, 0, 10])
                    obj.sleep()
            except:
                continue

            assert not np.any(np.isnan(target_obj.get_position()))
            try:
                ref_objs = self.scene.object_registry._objects_by_category[ref_obj_name]
                ref_obj = random.choice(list(ref_objs))
            except:
                continue
            all_ref.add(ref_obj)

            assert not np.any(np.isnan(target_obj.get_position()))

            try:
                ret = self.place(place_action, target_obj, ref_obj)
            except:
                ret = False
            assert not np.any(np.isnan(target_obj.get_position()))
            if ret:
                # print('success',target_obj.get_position())
                self._target_obj.append(target_obj)
                ref_obj_must_use.add(ref_obj)
                self.place_obj[target_obj_name] = {'target_obj': target_obj_name, 'action': place_action, 'ref_obj': ref_obj_name}
                self.success_place += 1

    def place(self, action, init_obj, ref_obj):
        if self.args.map_collect > 0:
            return

        if action == 'place_ontop':
            ret = init_obj.states[object_states.on_top.OnTop]._set_value(ref_obj, True)
        elif action == 'place_inside':
            ret = init_obj.states[object_states.inside.Inside]._set_value(ref_obj, True)
        elif action == 'place_under':
            ret = init_obj.states[object_states.under.Under]._set_value(ref_obj, True)
        elif action == 'place_nextto':
            ret = init_obj.states[object_states.next_to.NextTo]._set_value(ref_obj, True)
        else:
            raise Exception

        return ret

    def step(self, action, action_args=None):
        """
        action list: [move_forward,
                      move_backward,
                      turn_left,
                      turn_right,
                      look_up,
                      look_down,
                      open,
                      done,
                      leave,
                      test
                    ]
        Returns:
            obs: get the dict of rgbd, position and instructions
            reward: using the shortest distance of the objects
            done: 
            info: 
        """

        self.step_cnt += 1

        if not action == None:
            output = self.action_function[action](action_args)
        og.sim.my_step()
        og.sim.my_step()

        # Grab observations
        obs, info = self.agent.get_obs()
        rgb = obs['rgb']
        rgb = rgb[:, :, :3]
        depth = obs['depth_linear']
        # depth = 1 / (depth + 1e-10)
        # depth = depth / 100
        depth[depth > 10] = 10
        bbox_lst = get_bbox(obs, info)
        visible = [bbox['semanticLabel'] for bbox in bbox_lst]
        visible = set(visible)
        # depth_linear = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        agent_pos = self.agent.get_position()
        agent_ori = self.agent.get_orientation()
        agent_ori = agent_ori[[3, 0, 1, 2]]
        delta_position = agent_pos - self.current_episode_init_position
        delta_orientation = quat2euler(agent_ori)[2] - self.current_episode_init_orientation[2]
        gps_compass = delta_position + [delta_orientation]
        obs = {
            'rgb': rgb,
            'depth': depth,
            'seg_semantic': obs['seg_semantic'],
            'info': info,
            'pos': agent_pos,
            'ori': agent_ori,
            'instruction': self.now_task['user_instruction'],
            'habit_knowledge': self.habit_knowledge,
            'prompt': self.now_task['prompt'],
            'gps_compass': gps_compass,
            'task': self.now_task,
            'visible': visible,
            'bbox': bbox_lst
        }
        # if action == 'move_forward':
        #     obs['collision'] = not output
        #     if 'position' in action_args:
        #         self.scene.trav_map.floor_graph[0].remove_node(tuple(self.scene.trav_map.world_to_map(action_args['position'][:2])))
        # else:
        #     obs['collision'] = False
        # if action == 'turn_left' or action == 'turn_right':
        #     obs['turn_collision'] = not output
        # else:
        #     obs['turn_collision'] = False

        reward, done = self.reward(action, output)

        info = {}
        if action == 'done':
            info = {'find_success': True} if output else {'find_success': False}
        elif action == 'leave':
            info = {'leave_success': True} if output else {'leave_success': False}

        # Increment step
        self._current_step += 1

        return obs, reward, done, info

    # here are the action
    def done(self):
        pass

    def no_action(self):
        pass

    def move_forward(self, action_args=None):
        ori = self.agent.get_orientation()
        # ori = ori[[3, 0, 1, 2]]
        angle = quaternion_to_euler_angle(np.round(ori, 3))[0]
        angle += 1.5708
        if action_args is not None and 'step_size' in action_args:
            dx = action_args['step_size'] * np.cos(angle)
            dy = action_args['step_size'] * np.sin(angle)
        else:
            dx = self.step_size * np.cos(angle)
            dy = self.step_size * np.sin(angle)
        x, y, z = self.agent.get_position()
        if action_args is not None and 'position' in action_args:
            new_pos = np.round((action_args['position'][0], action_args['position'][1], z), 3)
        new_pos = np.round([x + dx, y + dy, z], 3)
        for xy in self.world_xy_list:
            if np.linalg.norm(np.array(new_pos)[:2] - np.array(xy)) < 0.01:
                self.agent.set_position((xy[0], xy[1], self.cam_height))
                if np.linalg.norm(self.agent.get_position()[:2] - new_pos[:2]) < 0.01:
                    return True
        return False

    def move_backward(self, action_args=None):
        ori = self.agent.get_orientation()
        angle = quaternion_to_euler_angle(np.round(ori, 3))[0]
        dx = self.step_size * np.cos(angle)
        dy = self.step_size * np.sin(angle)
        x, y, z = self.agent.get_position()
        new_pos = [x - dx, y - dy, z]
        for xy in self.world_xy_list:
            if np.linalg.norm(np.array(new_pos)[:2] - np.array(xy)) < 0.01:
                self.agent.set_position((xy[0], xy[1], self.cam_height))
                return True
        return False

    def turn_left(self, action_args=None):
        quat = self.agent.get_orientation()

        quat = quat[[3, 0, 1, 2]]
        # print(quaternion_to_euler_angle(quat))
        quat = qmult((euler2quat(0, 0, self.turn_size)), quat)
        # print(quaternion_to_euler_angle(quat))
        quat = quat[[1, 2, 3, 0]]
        self.agent.set_orientation(quat)
        ori = self.agent.get_orientation()
        for idx in range(len(ori)):
            ori[idx] = min(self.orientation_part_list, key=lambda x: abs(ori[idx] - x))
        self.agent.set_orientation(ori)

    def turn_right(self, action_args=None):
        quat = self.agent.get_orientation()
        quat = quat[[3, 0, 1, 2]]
        quat = qmult((euler2quat(0, 0, -self.turn_size)), quat)
        quat = quat[[1, 2, 3, 0]]
        self.agent.set_orientation(quat)
        ori = self.agent.get_orientation()
        for idx in range(len(ori)):
            ori[idx] = min(self.orientation_part_list, key=lambda x: abs(ori[idx] - x))
        self.agent.set_orientation(ori)

    def look_up(self, action_args=None):
        if self.cam_height < 1.3:
            self.cam_height += 0.5
        x, y, z = self.agent.get_position()
        self.agent.set_position((x, y, self.cam_height))

    def look_down(self, action_args=None):
        if self.cam_height > 0.7:
            self.cam_height -= 0.5
        x, y, z = self.agent.get_position()
        self.agent.set_position((x, y, self.cam_height))

    def open(self, action_args=None):
        openable_obj = []
        for obj in self.scene.objects:
            if not self.agent_near_obj(obj):
                continue
            if og.object_states.open_state.Open not in obj.states.keys():
                continue
            openable_obj.append(obj)
        for obj in openable_obj:
            ret = obj.states[og.object_states.open_state.Open]._set_value(True, fully=True)
            if not ret:
                ret = obj.states[og.object_states.open_state.Open]._set_value(True)

    def leave(self):
        pass

    def test(self):
        pass

    def done(self, action_args=None):
        """
        Returns:
        bool: whether the decision of done is accepted
        """
        obj = self.now_target
        agent_position = self.agent.get_position()
        obj_position = obj.get_position()
        # init_L2_dist = np.linalg.norm(np.array(self.agent_start_pos)[:2] - np.array(obj_position)[:2])
        # if init_L2_dist > 2:
        #     return False
        L2_distance = np.linalg.norm(np.array(agent_position)[:2] - np.array(obj_position)[:2])
        if self.agent_near_obj(self.now_target):
            obs, info = self.agent.get_obs()
            bbox_lst = get_bbox(obs, info)
            visible = [bbox['semanticLabel'] for bbox in bbox_lst]
            visible = set(visible)

            return self.now_target_name in visible

        return False

    # TODO
    def reward(self, action, accepted=None):
        return 0, False

    def place_agent(self):
        """
        set the pos of robot randomly

        Return:
            bool: whether the place action is success
        """
        world_xy_list = [tuple(xy) for xy in self.world_xy_list]
        xy = random.choice(world_xy_list)
        place_success = self.agent.set_position_orientation((xy[0], xy[1], self.init_cam_height), self.agent_start_ori)

        return place_success

    def place_agent_near_obj(self, obj, dist=2.0):
        obj_pos = obj.get_position()
        world_xy_list = [tuple(xy) for xy in self.world_xy_list]
        # world_xy_list = sorted(world_xy_list, key=lambda xy: np.abs(dist - np.sqrt((xy[0] - obj_pos[0])**2 + (xy[1] - obj_pos[1])**2)))
        world_xy_list = [xy for xy in self.world_xy_list if np.abs(dist - np.sqrt((xy[0] - obj_pos[0])**2 + (xy[1] - obj_pos[1])**2)) < 0.3]
        if world_xy_list == []:
            world_xy_list = sorted(self.world_xy_list, key=lambda xy: np.abs(dist - np.sqrt((xy[0] - obj_pos[0])**2 + (xy[1] - obj_pos[1])**2)))
            world_xy_list = world_xy_list[:5]
        xy = random.choice(world_xy_list)
        self.agent.set_position_orientation((xy[0], xy[1], self.init_cam_height), self.agent_start_ori)

        return np.linalg.norm(np.array(self.agent.get_position())[:2] - xy[:2]) < 0.1

    def agent_near_obj(self, obj, threshold=0.7):
        min_corner, max_corner = obj.aabb
        if min_corner[0]-threshold <= self.agent_pos[0] <= max_corner[0]+threshold \
            and min_corner[1]-threshold <= self.agent_pos[1] <= max_corner[1]+threshold:
            return True
        return False

    def restore_config_json(self, scene):

        dataset_path = gm.DATASET_PATH
        scene_path = os.path.join(dataset_path, 'scenes', scene)
        if self.id is None:
            used_file = f'{scene}_best.json'
        else:
            used_file = f'{scene}_best_{self.id}.json'

        with open(os.path.join(scene_path, 'json', f'{scene}_copy.json'), 'r') as f:
            data = json.load(f)

        with open(os.path.join(scene_path, 'json', used_file), 'w') as f:
            json.dump(data, f, indent=4)

    def _extract_categories(scene_json: dict) -> set:
        cats = set()
        init_info = scene_json.get("objects_info", {}).get("init_info", {})
        for obj_name, spec in init_info.items():
            try:
                cat = spec.get("args", {}).get("category", None)
                if cat:
                    cats.add(cat)
            except Exception:
                pass
        return cats

    # def improve_config_json(self, scene, ref_obj):

    #     dataset_path = gm.DATASET_PATH
    #     scene_path = os.path.join(dataset_path, 'scenes', scene)
    #     if self.id is None:
    #         used_file = f'{scene}_best.json'
    #     else:
    #         used_file = f'{scene}_best_{self.id}.json'

    #     with open(os.path.join(scene_path, 'json', used_file), 'r') as f:
    #         data = json.load(f)

    #     with open(os.path.join(scene_path, 'json', f'{scene}_copy.json'), 'w') as f:
    #         json.dump(data, f, indent=4)

    #     t = deepcopy(ref_obj)
    #     # print(data)
    #     for obj in t:
    #         if obj in data['category']:
    #             ref_obj.remove(obj)
    #         else:
    #             self.add_obj.append(obj)

    #     for filename in os.listdir(scene_path + '/json'):
    #         if filename == used_file:
    #             continue
    #         if len(ref_obj) == 0:
    #             break
    #         with open(os.path.join(scene_path, 'json', filename), 'r') as f:
    #             category = json.load(f)['category']
    #         t = deepcopy(ref_obj)
    #         for obj in t:
    #             flag = False
    #             if obj in category:
    #                 # ref_obj.remove(obj)
    #                 with open(os.path.join(scene_path, 'json', filename), 'r') as f:
    #                     new_data = json.load(f)
    #                 for obj_name in new_data['objects_info']['init_info']:
    #                     # try:
    #                     if new_data['objects_info']['init_info'][obj_name]['args']['name'] != 'robot0':
    #                         if new_data['objects_info']['init_info'][obj_name]['args']['category'] == obj:
    #                             data['state']['object_registry'][obj_name] = new_data['state']['object_registry'][obj_name]
    #                             data['objects_info']['init_info'][obj_name] = new_data['objects_info']['init_info'][obj_name]
    #                             flag = True
    #             if flag == True:
    #                 ref_obj.remove(obj)

    #     # assert len(ref_obj) == 0, '???'

    #     to_del = []
    #     for name in data['state']['object_registry']:
    #         if 'door_' in name:
    #             to_del.append(name)
    #     for name in to_del:
    #         del data['state']['object_registry'][name]
    #         del data['objects_info']['init_info'][name]

    #     with open(os.path.join(scene_path, 'json', used_file), 'w') as f:
    #         json.dump(data, f, indent=4)

    def improve_config_json(self, scene, ref_obj):
        def _extract_categories(scene_json: dict) -> set:
            """从 objects_info.init_info 里提取所有出现过的 category 名称。"""
            cats = set()
            init_info = scene_json.get("objects_info", {}).get("init_info", {})
            for obj_name, spec in init_info.items():
                try:
                    cat = spec.get("args", {}).get("category", None)
                    if cat:
                        cats.add(cat)
                except Exception:
                    pass
            return cats

        dataset_path = gm.DATASET_PATH
        scene_path = os.path.join(dataset_path, 'scenes', scene)
        json_dir = os.path.join(scene_path, 'json')

        if self.id is None:
            used_file = f'{scene}_best.json'
        else:
            used_file = f'{scene}_best_{self.id}.json'

        used_path = os.path.join(json_dir, used_file)
        copy_path = os.path.join(json_dir, f'{scene}_copy.json')

        # 1) 读取当前 best
        with open(used_path, 'r') as f:
            data = json.load(f)

        # 2) 把当前 best 备份成 copy（保留你原来的行为）
        with open(copy_path, 'w') as f:
            json.dump(data, f, indent=4)

        # ==== 改动 A：现有类别从 init_info 推导，而不是 data['category'] ====
        existing_cats = _extract_categories(data)

        t = deepcopy(ref_obj)
        for obj in t:
            if obj in existing_cats:
                ref_obj.remove(obj)
            else:
                self.add_obj.append(obj)

        # 3) 遍历同目录其他 json，补齐 ref_obj 里缺的类别
        for filename in os.listdir(json_dir):
            if filename == used_file:
                continue
            if len(ref_obj) == 0:
                break

            fp = os.path.join(json_dir, filename)
            try:
                with open(fp, 'r') as f:
                    new_data = json.load(f)
            except Exception:
                continue

            # ==== 改动 B：也用 init_info 推导这个 json 里有哪些类别 ====
            file_cats = _extract_categories(new_data)
            if not file_cats:
                continue

            t = deepcopy(ref_obj)
            for obj in t:
                if obj not in file_cats:
                    continue

                flag = False
                init_info = new_data.get('objects_info', {}).get('init_info', {})
                for obj_name, spec in init_info.items():
                    try:
                        # 保留你原来的 robot0 排除逻辑
                        if spec.get('args', {}).get('name') == 'robot0':
                            continue
                        if spec.get('args', {}).get('category') == obj:
                            data['state']['object_registry'][obj_name] = new_data['state']['object_registry'][obj_name]
                            data['objects_info']['init_info'][obj_name] = spec
                            flag = True
                    except Exception:
                        continue

                if flag:
                    ref_obj.remove(obj)
                    existing_cats.add(obj)

        # assert len(ref_obj) == 0, f"Missing categories not found in any json: {ref_obj}"

        # 4) 删掉所有 door_ 开头的 object（保留你原来的逻辑）
        to_del = []
        for name in data['state']['object_registry']:
            if 'door_' in name:
                to_del.append(name)
        for name in to_del:
            del data['state']['object_registry'][name]
            del data['objects_info']['init_info'][name]

        # ==== 可选但很有用：重建 data['category']，以后再用就不会 KeyError ====
        data['category'] = sorted(_extract_categories(data))

        # 5) 把修改后的 data 写回 used_file
        with open(used_path, 'w') as f:
            json.dump(data, f, indent=4)

    @property
    def target_pos(self):
        return self.now_target.get_position()

    @property
    def target_height(self):
        return self.target_pos[2]

    @property
    def agent_pos(self):
        return self.agent.get_position()

    @property
    def agent_orientation(self):
        return self.agent.get_orientation()

    def dist_2d(self, obj1, obj2):
        obj1_pos = obj1.get_position()
        obj2_pos = obj2.get_position()
        L2_distance = np.linalg.norm(np.array(obj1_pos)[:2] - np.array(obj2_pos)[:2])
        return L2_distance


def sample_unique_dicts(lst, k):
    random.shuffle(lst)
    name_to_dict = {d['basic_object_name']: d for i, d in enumerate(lst) if 'basic_object_name' in d}
    unique_names = set(name_to_dict.keys())
    if len(unique_names) < k:
        # assert len(unique_names) >= k, '???'
        k = len(unique_names)
    sampled_names = random.sample(unique_names, k)
    sampled_dicts = [name_to_dict[name] for name in sampled_names]

    return sampled_dicts


def get_ref_obj(obj_location):
    ref_obj = set()

    for location in obj_location:

        target_obj_name, rest = location.split('.')
        place_action, ref_obj_name = rest.split('(')
        ref_obj_name = ref_obj_name.replace(')', '')

        ref_obj.add(target_obj_name)
        ref_obj.add(ref_obj_name)

    return list(ref_obj)


def quaternion_to_euler_angle(q):
    w, x, y, z = q
    return p.getEulerFromQuaternion([x, y, z, w])


def euler_angle_to_quaternion(e):
    x, y, z, w = p.getQuaternionFromEuler(e)
    return [w, x, y, z]


def dist_2d(obj1, obj2):
    obj1_pos = obj1.get_position()
    obj2_pos = obj2.get_position()
    L2_distance = np.linalg.norm(np.array(obj1_pos)[:2] - np.array(obj2_pos)[:2])
    return L2_distance


def get_bbox(obs, info):
    """Compute per-instance bounding boxes and semantic labels without OpenCV.

    This is a drop-in replacement for the original cv2-based implementation.
    It uses seg_instance to find pixel extents, and samples any pixel inside
    the instance mask to query seg_semantic -> semanticLabel mapping from info.
    """
    unique_values = np.unique(obs['seg_instance'])
    bbox_dtype = np.dtype([
        ('x_min', '<i4'), ('y_min', '<i4'),
        ('x_max', '<i4'), ('y_max', '<i4'),
        ('semanticLabel', 'O')
    ])
    bbox_lst = []
    seg_inst = obs['seg_instance']
    seg_sem = obs['seg_semantic']
    sem_map = info.get('seg_semantic', {})

    for val in unique_values:
        if val == 0:
            continue
        ys, xs = np.where(seg_inst == val)
        if ys.size == 0:
            continue
        x_min = int(xs.min())
        x_max = int(xs.max()) + 1  # exclusive
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1  # exclusive

        # Pick a representative pixel for semantic id lookup
        semantic_id = int(seg_sem[ys[0], xs[0]])
        semantic_label = sem_map.get(semantic_id, None)

        bbox_lst.append((x_min, y_min, x_max, y_max, semantic_label))

    return np.array(bbox_lst, dtype=bbox_dtype)

class Map_Environment(Environment):

    def __init__(self,
                 configs,
                 id=None,
                 action_timestep=1 / 60.0,
                 physics_timestep=1 / 60.0,
                 device=None,
                 automatic_reset=False,
                 flatten_action_space=False,
                 flatten_obs_space=False,
                 block_size=3.0,
                 max_threshold=2.0,
                 min_threshold=0.5,
                 open_dist=0.7,
                 args=None):

        self.args = args
        self.init_angle = np.deg2rad(np.arange(0, 360, 90))
        self.init_quaternion = [p.getQuaternionFromEuler([angle, 0, 0]) for angle in self.init_angle]
        self.init_quaternion = [(x, y, z, w) for w, x, y, z in self.init_quaternion]
        self.step_size = args.step_size
        self.turn_size = args.turn_size
        self._task_path = args.task_path
        self._configs = configs
        self.choose_num = args.target_obj_num
        self.id = id
        self.change_task_cnt = 0
        self.max_step = args.max_step if 'max_step' in args else 50
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.open_dist = open_dist
        self.block_size = block_size
        self.look_angle = 0
        self.init_dist = 0
        self.leave_dist = 2.0
        self.find_dist = 1.0
        self.prev_dist = -1
        self.exploration_map = set()
        self.step_cnt = 0
        self.orientation_part_list = [-1, -np.sqrt(2) / 2, -1 / 2, 0, np.sqrt(2) / 2, 1 / 2, 1]
        self.init_cam_height = 1

        # for test
        self.place_cnt = 0
        self.success_place = 0

        with open(self._task_path, 'r') as f:
            task_data = json.load(f)
        scene_list = list(task_data.keys())
        scene = random.choice(scene_list)
        if args.scene != 'default':
            scene = args.scene
        self.scene_name = scene
        configs['scene']['scene_model'] = scene

        while True:
            # self.restore_config_json(scene)

            # the task used for initializing the env
            self._task_list = sample_unique_dicts(task_data[scene], self.choose_num)
            self._obj_location = [random.choice(task_dict['location']) for task_dict in self._task_list]
            self.place_obj = {}
            self.place_cnt += len(self._task_list)

            # modify the best_json for the task
            # ref_obj = get_ref_obj(self._obj_location)
            # self._ref_obj = deepcopy(ref_obj)
            # self.add_obj = []
            # self.improve_config_json(scene, ref_obj)

            try:
                og.sim.stop()
            except:
                pass
            super().__init__(configs=configs, action_timestep=action_timestep, physics_timestep=physics_timestep, device=device, automatic_reset=automatic_reset, flatten_action_space=flatten_action_space, flatten_obs_space=flatten_obs_space, id=self.id)
            og.sim.play()

            # self.init_robot_quaternion = self.robots[0].get_orientation()

            # place the target obj
            self._target_obj = []
            self.init_place()
            self._target_obj_num = len(self._target_obj)
            self._task_list = [task for task in self._task_list if task['basic_object_name'] in \
                [obj.category for obj in self._target_obj]]
            self.habit_knowledge = []
            for task in self._task_list:
                self.habit_knowledge.extend(task['habits'])
            random.shuffle(self.habit_knowledge)
            if len(self._task_list) > 0 or self.args.map_collect > 0:
                break
            else:
                pass

        self.state = og.sim.scene.dump_state(serialized=False)
        map_xy_list = list(self.scene.trav_map.floor_graph[0].nodes())
        self.world_xy_list = [self.scene.trav_map.map_to_world(np.array(xy)) for xy in map_xy_list]

        # self.now_task = random.choice(self._task_list)
        # self.now_target_name = self.now_task['location'][0].split('.')[0]
        # for target in self._target_obj:
        #     if target.category == self.now_target_name:
        #         self.now_target = target
        #         break

        # self._task_list.remove(self.now_task)

        self.agent = og.sim.viewer_camera
        ori = self.agent.get_orientation()
        for idx in range(len(ori)):
            ori[idx] = min(self.orientation_part_list, key=lambda x: abs(ori[idx] - x))
        self.agent.set_orientation(ori)
        self.agent_start_ori = self.agent.get_orientation()
        # self.place_agent()
        self.init_dist = random.choice([self.leave_dist, self.find_dist])
        # self.place_agent_near_obj(self.now_target, dist=self.init_dist)
        self.agent_start_pos = self.agent.get_position()
        self.agent_start_ori = self.agent.get_orientation()
        self.cam_height = self.init_cam_height

        # action space
        self.action_list = ['no_action', 'done', 'move_forward', 'move_backward'
                            'turn_left', 'turn_right', 'look_up', 'look_down', 'open', 'leave']
        self.action_function = {
            'done': self.done,
            'move_forward': self.move_forward,
            'move_backward': self.move_backward,
            'turn_left': self.turn_left,
            'turn_right': self.turn_right,
            'look_up': self.look_up,
            'look_down': self.look_down,
            'open': self.open,
            'leave': self.leave,
            'test': self.test,
            'no_action': self.no_action
        }

    def reset(self, change_scene=False):
        """
        Reset episode.
        """
        og.sim.load_state(self.state, serialized=False)
        self.prev_dist = -1
        self.exploration_map = set()

        with open(self._task_path, 'r') as f:
            task_data = json.load(f)

        self.max_step = 0
        # self.change_task_cnt += 1
        # if self.change_task_cnt > 1:
        #     self.step_cnt = 0
        #     self.change_task_cnt = 0
        #     no_change_scene = bool(len(self._task_list))
        #     if not no_change_scene:
        #         scene_list = list(task_data.keys())
        #         scene = self.scene_name
        #         self._configs['scene']['scene_model'] = self.scene_name

        #         while True:
        #             self.restore_config_json(scene)
        #             self._task_list = sample_unique_dicts(task_data[scene], self.choose_num)
        #             self._obj_location = [random.choice(task_dict['location']) for task_dict in self._task_list]
        #             self.place_cnt += len(self._task_list)

        #             # modify the best_json for the task
        #             ref_obj = get_ref_obj(self._obj_location)
        #             self._ref_obj = deepcopy(ref_obj)
        #             self.add_obj = []
        #             self.improve_config_json(scene, ref_obj)

        #             og.sim.stop()
        #             self.reload(configs=self._configs)
        #             og.sim.viewer_camera.SEMANTIC_REMAPPER.clear()
        #             og.sim.play()

        #             # place the target obj
        #             self._target_obj = []
        #             self.init_place()
        #             self._target_obj_num = len(self._target_obj)
        #             self._task_list = [task for task in self._task_list if task['basic_object_name'] \
        #                  in [obj.category for obj in self._target_obj]]
        #             self.habit_knowledge = []
        #             for task in self._task_list:
        #                 self.habit_knowledge.extend(task['habits'])
        #             random.shuffle(self.habit_knowledge)
        #             if len(self._task_list) > 0 or self.args.map_collect > 0:
        #                 break
        #             else:
        #                 pass

        #         print(self.success_place / self.place_cnt)

        #         map_xy_list = list(self.scene.trav_map.floor_graph[0].nodes())
        #         self.world_xy_list = [self.scene.trav_map.map_to_world(np.array(xy)) for xy in map_xy_list]

        #         # set sleep to disable physics
        #         for obj in self.scene.objects:
        #             if og.object_states.open_state.Open not in obj.states.keys():
        #                 obj.sleep()

        #     self.now_task = random.choice(self._task_list)
        #     self.now_target_name = self.now_task['location'][0].split('.')[0]
        #     for target in self._target_obj:
        #         if target.category == self.now_target_name:
        #             self.now_target = target
        #             break
        #     self._task_list.remove(self.now_task)

        for obj in self.scene.objects:
            try:
                obj.get_position()
            except:
                og.sim.stop()
                og.sim.play()
                print('refresh')
            if np.any(np.isnan(obj.get_position())):
                og.sim.stop()
                og.sim.play()
                print('refresh')
        self.state = og.sim.scene.dump_state(serialized=False)

        # keep the agent new
        self.agent = og.sim.viewer_camera
        for i in range(random.choice([0, 1, 2, 3])):
            self.turn_left()

        # keep the agent new
        self.agent = og.sim.viewer_camera
        ori = self.agent.get_orientation()
        for idx in range(len(ori)):
            ori[idx] = min(self.orientation_part_list, key=lambda x: abs(ori[idx] - x))
        self.agent.set_orientation(ori)
        self.agent_start_ori = self.agent.get_orientation()
        # self.place_agent()
        # self.init_dist = random.choice([self.leave_dist, self.find_dist])
        # self.place_agent_near_obj(self.now_target, dist=self.init_dist)
        # if self.args.map_collect > 0:
        #     self.place_agent_zero()
        # else:
        #     self.place_agent()

        self.agent_start_pos = self.agent.get_position()
        self.agent_start_ori = self.agent.get_orientation()
        self.exploration_map = set()
        self.look_angle = 0
        self.success_reward_sum, self.dist_reward_sum, self.exploration_reward_sum, self.block_penalty_sum = 0, 0, 0, 0

        # Reset internal variables
        super()._reset_variables()

        # Run a single simulator step to make sure we can grab updated observations
        og.sim.my_step()
        og.sim.my_step()

        # Grab observations
        obs, info = self.agent.get_obs()
        rgb = obs['rgb']
        rgb = rgb[:, :, :3]
        depth = obs['depth_linear']
        # depth = 1 / (depth + 1e-10)
        # depth = depth / 100
        depth[depth > 10] = 10
        bbox_lst = get_bbox(obs, info)
        visible = [bbox['semanticLabel'] for bbox in bbox_lst]
        visible = set(visible)
        # depth_linear = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        agent_pos = self.agent.get_position()
        agent_ori = self.agent.get_orientation()
        agent_ori = agent_ori[[3, 0, 1, 2]]
        # agent_ori = quaternion.quaternion(*agent_ori)
        self.current_episode_init_position = agent_pos
        self.current_episode_init_orientation = quat2euler(agent_ori)
        gps_compass = np.zeros(3)
        # obs = {'rgb': rgb, 'depth': depth, 'pos': agent_pos, 'ori': agent_ori, 'instruction': self.now_task['user_instruction'], 'prompt': self.now_task['prompt'], 'gps_compass': gps_compass,'task':self.now_task, 'visible':visible, 'location': self.now_location }
        obs = {'rgb': rgb, 'depth': depth, 'pos': agent_pos, 'ori': agent_ori, 'gps_compass': gps_compass, 'visible': visible, 'collision': False, 'turn_collision': False}

        return obs

    def place_agent_zero(self):
        land_success = self.agent.set_pos_ori_with_no_collision((0, 0, self.init_cam_height), quat=self.init_quaternion[0])
        return land_success

    def load(self):
        """
        Load the scene and robot specified in the config file.
        """
        # This environment is not loaded
        self._loaded = False

        # Load config variables
        self._load_variables()

        # Load the scene, robots, and task
        self._load_scene()
        # self._load_robots()
        self._load_objects()
        self._load_task()

        og.sim.play()
        self.task.reset(self)
        self._reset_variables()
        og.sim.my_step()
        og.sim.my_step()

        # Load the obs / action spaces
        self.load_observation_space()
        self._load_action_space()

        # Denote that the scene is loaded
        self._loaded = True

    def close(self):
        """
        restore the best_json
        """
        super().close()

        self.restore_config_json(self.scene_name)

    def init_place(self):
        """
        set the initial place of target objects generated by gpt4
        """
        if self.args.map_collect > 0:
            return

        self.place_obj = {}
        self._target_obj = []
        all_ref, ref_obj_must_use = set(), set()

        all_num, cnt = 0, 0
        for location in self._obj_location:

            for obj in self.scene.objects:
                try:
                    obj.get_position()
                except:
                    og.sim.stop()
                    og.sim.play()
                    print('refresh')
                if np.any(np.isnan(obj.get_position())):
                    og.sim.stop()
                    og.sim.play()
                    print('refresh')

            target_obj_name, rest = location.split('.')
            place_action, ref_obj_name = rest.split('(')
            ref_obj_name = ref_obj_name.replace(')', '')
            # choose the object to use
            target_obj, ref_obj = None, None

            # if target_obj_name.split('_')[-1].isdigit():
            #     target_obj = self.scene.object_registry._objects_by_name[target_obj_name]
            # else:
            to_remove = []
            try:
                target_objs = self.scene.object_registry._objects_by_category[target_obj_name]
                target_obj = random.choice(list(target_objs))
                for obj in target_objs:
                    if obj != target_obj:
                        to_remove.append(obj)
                for obj in to_remove:
                    # self.scene.remove_object(obj)
                    obj.set_position([0, 0, 10])
                    obj.sleep()
            except:
                continue

            assert not np.any(np.isnan(target_obj.get_position()))
            try:
                ref_objs = self.scene.object_registry._objects_by_category[ref_obj_name]
                ref_obj = random.choice(list(ref_objs))
            except:
                continue
            all_ref.add(ref_obj)

            assert not np.any(np.isnan(target_obj.get_position()))

            try:
                ret = self.place(place_action, target_obj, ref_obj)
            except:
                ret = False
            assert not np.any(np.isnan(target_obj.get_position()))
            if ret:
                # print('success',target_obj.get_position())
                self._target_obj.append(target_obj)
                ref_obj_must_use.add(ref_obj)
                self.place_obj[target_obj_name] = {'target_obj': target_obj_name, 'action': place_action, 'ref_obj': ref_obj_name}
                self.success_place += 1

    def place(self, action, init_obj, ref_obj):
        if self.args.map_collect > 0:
            return

        if action == 'place_ontop':
            ret = init_obj.states[object_states.on_top.OnTop]._set_value(ref_obj, True)
        elif action == 'place_inside':
            ret = init_obj.states[object_states.inside.Inside]._set_value(ref_obj, True)
        elif action == 'place_under':
            ret = init_obj.states[object_states.under.Under]._set_value(ref_obj, True)
        elif action == 'place_nextto':
            ret = init_obj.states[object_states.next_to.NextTo]._set_value(ref_obj, True)
        else:
            raise Exception

        return ret

    def step(self, action, action_args=None):
        """
        action list: [move_forward,
                      move_backward,
                      turn_left,
                      turn_right,
                      look_up,
                      look_down,
                      open,
                      done,
                      leave,
                      test
                    ]
        Returns:
            obs: get the dict of rgbd, position and instructions
            reward: using the shortest distance of the objects
            done: 
            info: 
        """

        self.step_cnt += 1

        if not action == None:
            output = self.action_function[action](action_args)
        for _ in range(2):
            og.sim.my_step()

        # Grab observations
        obs, info = self.agent.get_obs()
        rgb = obs['rgb']
        rgb = rgb[:, :, :3]
        depth = obs['depth_linear']
        # depth = 1 / (depth + 1e-10)
        # depth = depth / 100
        depth[depth > 10] = 10
        bbox_lst = get_bbox(obs, info)
        visible = [bbox['semanticLabel'] for bbox in bbox_lst]
        visible = set(visible)
        # depth_linear = obs['robot0']['robot0:eyes_Camera_sensor_depth_linear']
        agent_pos = self.agent.get_position()
        agent_ori = self.agent.get_orientation()
        agent_ori = agent_ori[[3, 0, 1, 2]]
        delta_position = agent_pos - self.current_episode_init_position
        delta_orientation = quat2euler(agent_ori)[2] - self.current_episode_init_orientation[2]
        gps_compass = delta_position + [delta_orientation]
        obs = {'rgb': rgb, 'depth': depth, 'pos': agent_pos, 'ori': agent_ori, 'gps_compass': gps_compass, 'visible': visible}
        if action == 'move_forward':
            obs['collision'] = not output
            if 'position' in action_args:
                self.scene.trav_map.floor_graph[0].remove_node(tuple(self.scene.trav_map.world_to_map(action_args['position'][:2])))
        else:
            obs['collision'] = False
        if action == 'turn_left' or action == 'turn_right':
            obs['turn_collision'] = not output
        else:
            obs['turn_collision'] = False

        reward, done = self.reward(action, output)

        info = {}
        if action == 'done':
            info = {'find_success': True} if output else {'find_success': False}
        elif action == 'leave':
            info = {'leave_success': True} if output else {'leave_success': False}

        # Increment step
        self._current_step += 1

        return obs, reward, done, info

    # here are the action

    def move_forward(self, action_args=None):
        ori = self.agent.get_orientation()
        angle = quaternion_to_euler_angle(np.round(ori, 3))[0]
        angle += 1.5708
        if action_args is not None and 'step_size' in action_args:
            dx = action_args['step_size'] * np.cos(angle)
            dy = action_args['step_size'] * np.sin(angle)
        else:
            dx = self.step_size * np.cos(angle)
            dy = self.step_size * np.sin(angle)
        x, y, z = self.agent.get_position()
        if action_args is not None and 'position' in action_args:
            new_pos = np.round((action_args['position'][0], action_args['position'][1], z), 2)
        new_pos = np.round([x + dx, y + dy, z], 3)
        for xy in self.world_xy_list:
            if np.linalg.norm(np.array(new_pos)[:2] - np.array(xy)) < 0.01:
                self.agent.set_position((xy[0], xy[1], self.cam_height))
                if np.linalg.norm(self.agent.get_position()[:2] - new_pos[:2]) < 0.01:
                    return True
        return False

    def move_backward(self, action_args=None):
        ori = self.agent.get_orientation()
        angle = quaternion_to_euler_angle(np.round(ori, 3))[0]
        dx = self.step_size * np.cos(angle)
        dy = self.step_size * np.sin(angle)
        x, y, z = self.agent.get_position()
        new_pos = [x - dx, y - dy, z]
        for xy in self.world_xy_list:
            if np.linalg.norm(np.array(new_pos)[:2] - np.array(xy)) < 0.01:
                self.agent.set_position((xy[0], xy[1], self.cam_height))
                return True
        return False

    def turn_left(self, action_args=None):
        quat = self.agent.get_orientation()
        quat = quat[[3, 0, 1, 2]]
        quat = qmult((euler2quat(0, 0, self.turn_size)), quat)
        quat = quat[[1, 2, 3, 0]]
        self.agent.set_orientation(quat)
        ori = self.agent.get_orientation()
        for idx in range(len(ori)):
            ori[idx] = min(self.orientation_part_list, key=lambda x: abs(ori[idx] - x))
        self.agent.set_orientation(ori)

    def turn_right(self, action_args=None):
        quat = self.agent.get_orientation()
        quat = quat[[3, 0, 1, 2]]
        quat = qmult((euler2quat(0, 0, -self.turn_size)), quat)
        quat = quat[[1, 2, 3, 0]]
        self.agent.set_orientation(quat)
        ori = self.agent.get_orientation()
        for idx in range(len(ori)):
            ori[idx] = min(self.orientation_part_list, key=lambda x: abs(ori[idx] - x))
        self.agent.set_orientation(ori)

    def look_up(self, action_args=None):
        if self.cam_height < 1.3:
            self.cam_height += 0.5

    def look_down(self, action_args=None):
        if self.cam_height > 0.7:
            self.cam_height -= 0.5

    def open(self, action_args=None):
        openable_obj = []
        for obj in self.scene.objects:
            if not self.agent_near_obj(obj):
                continue
            if og.object_states.open_state.Open not in obj.states.keys():
                continue
            openable_obj.append(obj)
        for obj in openable_obj:
            ret = obj.states[og.object_states.open_state.Open]._set_value(True, fully=True)
            if not ret:
                ret = obj.states[og.object_states.open_state.Open]._set_value(True)

    def leave(self):
        pass

    def test(self):
        pass

    def done(self):
        """
        Returns:
        bool: whether the decision of done is accepted
        """
        obj = self.now_target
        agent_position = self.agent.get_position()
        obj_position = obj.get_position()
        # init_L2_dist = np.linalg.norm(np.array(self.agent_start_pos)[:2] - np.array(obj_position)[:2])
        # if init_L2_dist > 2:
        #     return False
        L2_distance = np.linalg.norm(np.array(agent_position)[:2] - np.array(obj_position)[:2])
        if self.agent_near_obj(self.now_target):
            obs, info = self.agent.get_obs()
            bbox_lst = get_bbox(obs, info)
            visible = [bbox['semanticLabel'] for bbox in bbox_lst]
            visible = set(visible)

            return self.now_target_name in visible

        return False

    # TODO
    def reward(self, action, accepted=None):
        return 0, False

    def place_agent(self):
        """
        set the pos of robot randomly

        Return:
            bool: whether the place action is success
        """
        world_xy_list = [tuple(xy) for xy in self.world_xy_list]
        xy = random.choice(world_xy_list)
        place_success = self.agent.set_position_orientation((xy[0], xy[1], self.init_cam_height), self.agent_start_ori)

        return place_success

    def place_agent_near_obj(self, obj, dist=2.0):
        obj_pos = obj.get_position()
        world_xy_list = [tuple(xy) for xy in self.world_xy_list]
        # world_xy_list = sorted(world_xy_list, key=lambda xy: np.abs(dist - np.sqrt((xy[0] - obj_pos[0])**2 + (xy[1] - obj_pos[1])**2)))
        world_xy_list = [xy for xy in self.world_xy_list if np.abs(dist - np.sqrt((xy[0] - obj_pos[0])**2 + (xy[1] - obj_pos[1])**2)) < 0.3]
        if world_xy_list == []:
            world_xy_list = sorted(self.world_xy_list, key=lambda xy: np.abs(dist - np.sqrt((xy[0] - obj_pos[0])**2 + (xy[1] - obj_pos[1])**2)))
            world_xy_list = world_xy_list[:5]
        xy = random.choice(world_xy_list)
        self.agent.set_position_orientation((xy[0], xy[1], self.init_cam_height), self.agent_start_ori)

        return np.linalg.norm(np.array(self.agent.get_position())[:2] - xy[:2]) < 0.1

    def agent_near_obj(self, obj, threshold=0.7):
        min_corner, max_corner = obj.aabb
        if min_corner[0]-threshold <= self.agent_pos[0] <= max_corner[0]+threshold \
            and min_corner[1]-threshold <= self.agent_pos[1] <= max_corner[1]+threshold:
            return True
        return False

    def restore_config_json(self, scene):

        dataset_path = gm.DATASET_PATH
        scene_path = os.path.join(dataset_path, 'scenes', scene)
        if self.id is None:
            used_file = f'{scene}_best.json'
        else:
            used_file = f'{scene}_best_{self.id}.json'

        with open(os.path.join(scene_path, 'json', f'{scene}_copy.json'), 'r') as f:
            data = json.load(f)

        with open(os.path.join(scene_path, 'json', used_file), 'w') as f:
            json.dump(data, f, indent=4)

    def improve_config_json(self, scene, ref_obj):

        dataset_path = gm.DATASET_PATH
        scene_path = os.path.join(dataset_path, 'scenes', scene)
        if self.id is None:
            used_file = f'{scene}_best.json'
        else:
            used_file = f'{scene}_best_{self.id}.json'

        with open(os.path.join(scene_path, 'json', used_file), 'r') as f:
            data = json.load(f)

        with open(os.path.join(scene_path, 'json', f'{scene}_copy.json'), 'w') as f:
            json.dump(data, f, indent=4)

        t = deepcopy(ref_obj)
        for obj in t:
            if obj in data['category']:
                ref_obj.remove(obj)
            else:
                self.add_obj.append(obj)

        for filename in os.listdir(scene_path + '/json'):
            if filename == used_file:
                continue
            if len(ref_obj) == 0:
                break
            with open(os.path.join(scene_path, 'json', filename), 'r') as f:
                category = json.load(f)['category']
            t = deepcopy(ref_obj)
            for obj in t:
                flag = False
                if obj in category:
                    # ref_obj.remove(obj)
                    with open(os.path.join(scene_path, 'json', filename), 'r') as f:
                        new_data = json.load(f)
                    for obj_name in new_data['objects_info']['init_info']:
                        # try:
                        if new_data['objects_info']['init_info'][obj_name]['args']['name'] != 'robot0':
                            if new_data['objects_info']['init_info'][obj_name]['args']['category'] == obj:
                                data['state']['object_registry'][obj_name] = new_data['state']['object_registry'][obj_name]
                                data['objects_info']['init_info'][obj_name] = new_data['objects_info']['init_info'][obj_name]
                                flag = True
                if flag == True:
                    ref_obj.remove(obj)

        assert len(ref_obj) == 0, '???'

        to_del = []
        for name in data['state']['object_registry']:
            if 'door_' in name:
                to_del.append(name)
        for name in to_del:
            del data['state']['object_registry'][name]
            del data['objects_info']['init_info'][name]

        with open(os.path.join(scene_path, 'json', used_file), 'w') as f:
            json.dump(data, f, indent=4)

    @property
    def target_pos(self):
        return self.now_target.get_position()

    @property
    def target_height(self):
        return self.target_pos[2]

    @property
    def agent_pos(self):
        return self.agent.get_position()

    @property
    def agent_orientation(self):
        return self.agent.get_orientation()

    def dist_2d(self, obj1, obj2):
        obj1_pos = obj1.get_position()
        obj2_pos = obj2.get_position()
        L2_distance = np.linalg.norm(np.array(obj1_pos)[:2] - np.array(obj2_pos)[:2])
        return L2_distance


def sample_unique_dicts(lst, k):
    random.shuffle(lst)
    name_to_dict = {d['basic_object_name']: d for i, d in enumerate(lst) if 'basic_object_name' in d}
    unique_names = set(name_to_dict.keys())
    if len(unique_names) < k:
        # assert len(unique_names) >= k, '???'
        k = len(unique_names)
    sampled_names = random.sample(unique_names, k)
    sampled_dicts = [name_to_dict[name] for name in sampled_names]
    # sampled_dict = []
    # chosen_object_name = random.sample(list(lst.keys()), min(k, len(lst.keys())))
    # for object_name in chosen_object_name:
    #     sampled_dict.append(random.choice(lst[object_name]))
    return sampled_dicts


def get_ref_obj(obj_location):
    ref_obj = set()

    for location in obj_location:

        target_obj_name, rest = location.split('.')
        place_action, ref_obj_name = rest.split('(')
        ref_obj_name = ref_obj_name.replace(')', '')

        ref_obj.add(target_obj_name)
        ref_obj.add(ref_obj_name)

    return list(ref_obj)


def quaternion_to_euler_angle(q):
    w, x, y, z = q
    return p.getEulerFromQuaternion([x, y, z, w])


def euler_angle_to_quaternion(e):
    x, y, z, w = p.getQuaternionFromEuler(e)
    return [w, x, y, z]


def dist_2d(obj1, obj2):
    obj1_pos = obj1.get_position()
    obj2_pos = obj2.get_position()
    L2_distance = np.linalg.norm(np.array(obj1_pos)[:2] - np.array(obj2_pos)[:2])
    return L2_distance


def get_bbox(obs, info):
    """Compute per-instance bounding boxes and semantic labels without OpenCV.

    This is a drop-in replacement for the original cv2-based implementation.
    It uses seg_instance to find pixel extents, and samples any pixel inside
    the instance mask to query seg_semantic -> semanticLabel mapping from info.
    """
    unique_values = np.unique(obs['seg_instance'])
    bbox_dtype = np.dtype([
        ('x_min', '<i4'), ('y_min', '<i4'),
        ('x_max', '<i4'), ('y_max', '<i4'),
        ('semanticLabel', 'O')
    ])
    bbox_lst = []
    seg_inst = obs['seg_instance']
    seg_sem = obs['seg_semantic']
    sem_map = info.get('seg_semantic', {})

    for val in unique_values:
        if val == 0:
            continue
        ys, xs = np.where(seg_inst == val)
        if ys.size == 0:
            continue
        x_min = int(xs.min())
        x_max = int(xs.max()) + 1  # exclusive
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1  # exclusive

        # Pick a representative pixel for semantic id lookup
        semantic_id = int(seg_sem[ys[0], xs[0]])
        semantic_label = sem_map.get(semantic_id, None)

        bbox_lst.append((x_min, y_min, x_max, y_max, semantic_label))

    return np.array(bbox_lst, dtype=bbox_dtype)