import ray
import ray.tune as tune
from ray.rllib import rollout
from ray.tune.registry import get_trainable_cls

import gym
import minerl
import minerl.env.core
import minerl.env.comms

import numpy as np
import argparse
import shutil
import random
import time

class MineRLEnvWrap(minerl.env.core.MineRLEnv):
    def __init__(self, xml, height, width):
        super().__init__(
            xml,
            gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
            gym.spaces.Discrete(3),
            None,
            port=9000
        )

    def _setup_spaces(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def _process_action(self, action_in) -> str:
        # regardless of settings (allow-list, deny-list) in xml,
        # this overwrites a list of available actions.
        action_to_command_array = [
            'move 1',
            'camera 0 90',  # right turn
            'camera 0 -90']   # left turn
        return action_to_command_array[action_in]

    def _process_observation(self, pov, info):
        # we need only pov to analyze
        pov = np.frombuffer(pov, dtype=np.uint8)
        pov = pov.reshape((self.height, self.width, self.depth))
        return pov

    def reset(self):
        self.action_history = []
        return super().reset()

    def step(self, action):
        self.action_history.append(action)

        o, r, d, i = super().step(action)
        return o, r, d, i

def create_env(config):
    mission = config["mission"]
    env = MineRLEnvWrap(mission, 84, 84)
    return env

def replace_placeholder_in_mission(filename, r_width, r_height, r_seed):
    with open(filename) as f_r:
        content = f_r.read().format(
            PLACEHOLDER_WIDTH=r_width,
            PLACEHOLDER_HEIGHT=r_height,
            PLACEHOLDER_SEED=r_seed)
    with open(filename, 'w') as f_w:
        f_w.write(content)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', required=False, default="./checkpoint/checkpoint-645")
    args = parser.parse_args()

    ray.init()

    run = True
    while run:

        #
        # Generate a seed for maze
        #
        print("Generating new seed ...")
        seed_maze = random.randint(1, 9999)

        #
        # run agent with trained checkpoint
        #

        # create mission file (84 x 84)
        shutil.copyfile("lava_maze_minerl-PLACEHOLDER.xml", "lava_maze_minerl-84x84.xml")
        replace_placeholder_in_mission(
            "lava_maze_minerl-84x84.xml",
            84,
            84,
            seed_maze)

        # start agent !
        print("An agent is running ...")
        tune.register_env("testenv01", create_env)
        cls = get_trainable_cls("DQN")
        config={
            #"log_level": "DEBUG",
            "env_config": {
                "mission": "lava_maze_minerl-84x84.xml"
            },
            "num_gpus": 0,
            "num_workers": 0,
            "double_q": True,
            "dueling": True,
            "explore": False
        }
        agent = cls(env="testenv01", config=config)
        agent.optimizer.stop()
        agent.restore(args.checkpoint_file)
        env1 = agent.workers.local_worker().env
        obs = env1.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.compute_action(obs)
            obs, reward, done, info = env1.step(action)
            total_reward += reward
        env1.close()
        agent.stop()
        print("Done with reward ", total_reward)

        #
        # simulate result
        #
        
        # create mission file (800 x 600)
        shutil.copyfile("lava_maze_minerl-PLACEHOLDER.xml", "lava_maze_minerl-800x600.xml")
        replace_placeholder_in_mission(
            "lava_maze_minerl-800x600.xml",
            800,
            600,
            seed_maze)

        # run simulation
        env2 = MineRLEnvWrap("lava_maze_minerl-800x600.xml", 600, 800)
        env2.reset()
        print("The world is loaded. Press F5 key for third-person view.")
        input("Enter keyboard to start simulation")
        for action in env1.action_history:
            time.sleep(0.5)
            obs, reward, done, info = env2.step(action)
        user_choice = input("Enter 'N' to exit [Y/n]: ").lower()
        if user_choice in ['n', 'no']:
            run = False
        env2.close()

    ray.shutdown()
