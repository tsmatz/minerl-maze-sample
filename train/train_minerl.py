# Reinforcement Learning with MineRL (MineRLEnv) 0.3.6 
import gym
import ray
import ray.tune as tune
import minerl.env.core
import minerl.env.comms
import numpy as np

class MineRLEnvWrap(minerl.env.core.MineRLEnv):
    def __init__(self, xml):
        super().__init__(
            xml,
            gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            gym.spaces.Discrete(3),
            None
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

def create_env(config):
    # for quick recovery
    minerl.env.core.SOCKTIME = 25.0 # default 60.0 * 4
    minerl.env.comms.retry_timeout = 1 # default 10
    minerl.env.comms.retry_count = 5 # default 20

    mission = config["mission"]
    env = MineRLEnvWrap(mission)
    return env

def stop_check(trial_id, result):
    return result["episode_reward_mean"] >= 85

if __name__ == '__main__':
    tune.register_env("testenv01", create_env)

    ray.init()

    tune.run(
        run_or_experiment="DQN",
        config={
            #"log_level": "DEBUG",
            "env": "testenv01",
            "env_config": {
                "mission": "./lava_maze_minerl.xml"
            },
            "num_gpus": 0,
            "num_workers": 1,
            "ignore_worker_failures": True,
            "double_q": True,
            "dueling": True,
            "explore": True,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 500000
            }
        },
        stop=stop_check,
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir='./logs'
    )

    print('training has done !')
    ray.shutdown()
