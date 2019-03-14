""" Contains the Episodes for Navigation. """
import random
import torch
import time
import sys
from constants import GOAL_SUCCESS_REWARD, STEP_PENALTY, BASIC_ACTIONS, GOAL2_SUCCESS_REWARD    # YZ
from environment import Environment
from utils.net_util import gpuify


class Episode:
    """ Episode for Navigation. """
    def __init__(self, args, gpu_id, rank, strict_done=False):
        super(Episode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None

        self.seed = args.seed + rank
        random.seed(self.seed)

        with open('./datasets/objects/int_objects.txt') as f:
            int_objects = [s.strip() for s in f.readlines()]
        with open('./datasets/objects/rec_objects.txt') as f:
            rec_objects = [s.strip() for s in f.readlines()]
        
        self.objects = int_objects + rec_objects

        self.actions_list = [{'action':a} for a in BASIC_ACTIONS]
        self.actions_taken = []


    @property
    def environment(self):
        return self._env

    def state_for_agent(self):
        # YZ: add network-found memory into the state space.
        memory = [0.0, 0.0]
        for i in range(len(self.target)):
            if self.target[i] in self.targets_found:
                memory[i] = 1.0
        return [self.environment.current_frame, memory]

    def step(self, action_as_int):
        action = self.actions_list[action_as_int]
        self.actions_taken.append(action)
        return self.action_step(action)

    def action_step(self, action):
        self.environment.step(action)
        reward, terminal, action_was_successful = self.judge(action)

        return reward, terminal, action_was_successful

    def slow_replay(self, delay=0.2):
        # Reset the episode
        self._env.reset(self.cur_scene, change_seed = False)
        
        for action in self.actions_taken:
            self.action_step(action)
            time.sleep(delay)
    
    def judge(self, action):
        """ Judge the last event. """
        # immediate reward
        reward = STEP_PENALTY 
        done = False
        action_was_successful = self.environment.last_action_success

        if action['action'] in range(len(self.target)):
            target_index = action['action']
            # YZ: add (what network thinks) found target in memory
            if self.target[target_index] not in self.targets_found:
                self.targets_found.append(self.target[target_index])

            # YZ: add (what is really) found target in memory
            objects = self._env.last_event.metadata['objects']
            visible_objects = [o['objectType'] for o in objects if o['visible']]
            # YZ-comment: visible objects in current frame of the environment, eg. ['Cup','Bowl']
            if self.target[target_index] in visible_objects:
                if self.target[target_index] not in self.targets_done:
                    self.targets_done.append(self.target[target_index])
                    if len(self.targets_done) == 1:
                        reward += GOAL_SUCCESS_REWARD
                    elif len(self.targets_done) == 2:
                        reward += GOAL2_SUCCESS_REWARD

            # YZ: if all targets have been network-found
            if len(self.target) == len(self.targets_found):
                done = True
            # YZ: if all targets have been really-found
            if len(self.target) == len(self.targets_done):
                self.success = True

        return reward, done, action_was_successful

    def new_episode(self, args, scene):
        
        if self._env is None:
            if args.arch == 'osx':
                local_executable_path = './datasets/builds/thor-local-OSXIntel64.app/Contents/MacOS/thor-local-OSXIntel64'
            else:
                local_executable_path = './datasets/builds/thor-local-Linux64'
            
            self._env = Environment(
                    grid_size=args.grid_size,
                    fov=args.fov,
                    local_executable_path=local_executable_path,
                    randomize_objects=args.randomize_objects,
                    seed=self.seed)
            self._env.start(scene, self.gpu_id)
        else:
            self._env.reset(scene)

        # For now, single target.
        self.target = ['Tomato', 'Bowl']    # YZ-revised from: self.target = 'Tomato'

        self.success = False
        self.cur_scene = scene
        self.actions_taken = []

        self.targets_found = []     # YZ-add memory for network-found targets within current episode.
        self.targets_done = []      # YZ-add memory for really-found targets within current episode.

        return True
