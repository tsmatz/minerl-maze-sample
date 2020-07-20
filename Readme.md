# Reinforcement Learning for Minecraft (MineRL) Sample

This sample code trains an agent in Minecraft with reinforcement learning.    
In this example, an agent will learn to reach to a goal block in a given maze.

**Do not use low-spec machine**, since the training worker will request enough resources.    
(Here I used Ubuntu 18.04 on Standard D3 v2 with 4 cores and 14 GB RAM in Microsoft Azure.)

See my post "[Enjoy AI in Minecraft (Malmo and MineRL)](https://tsmatz.wordpress.com/2020/07/09/minerl-and-malmo-reinforcement-learning-in-minecraft/)" for details and background.

## 1. Setup prerequisite environment

We assume Ubuntu 18.04 in this tutorial.    
First, make sure that python 3 is installed. (If not, please install python 3.x.)

```
python3 -V
```

Install ```pip```.

```
sudo apt-get update
sudo apt-get -y install python3-pip
sudo -H pip3 install --upgrade pip
```

MineRL requires a monitor (screen) to run Minecraft.    
Then, install and start service for remote desktop (such as, VNC).

```
sudo apt-get update
sudo apt-get install lxde -y
sudo apt-get install xrdp -y
/etc/init.d/xrdp start  # password is required
```

> Use a virtual monitor (such as, ```xvfb```) instead, when you run training as a batch. (For simplicity, we use a real monitor in this example.)

Allow inbound port 3389 (remote desktop protocol) for this computer in network setting.

Restart your computer.

Finally, install Java runtime and set ```JAVA_HOME```, since Minecarft client with mods (Java version) is used.

```
sudo apt-get install openjdk-8-jdk
echo -e "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64" >> ~/.bashrc
source ~/.bashrc
```

## 2. Install MineRL

```
pip3 install minerl==0.3.6
```

## 3. Install Ray and RLlib framework

```
pip3 install pandas tensorflow==1.15 tensorboardX tabulate dm_tree lz4 ray==0.8.3 ray[rllib]==0.8.3 ray[tune]==0.8.3
```

## 4. Run a training (reinforcement learning)

Login to your machine with remote desktop (with a monitor).

When it launches 2 instances of Minecraft client, please close one, which is not used for training. (Ray will create a dummy env in order to see settings in environment, such as, action space or observation space.)

```
cd train
python3 train_minerl.py
```

> This training will take so long time, since it runs on a single instance. To run distributed training, configure ray cluster (multiple workers). (You can quickly configure using [Azure Machine Learning](https://tsmatz.wordpress.com/2018/11/20/azure-machine-learning-services/) on cloud.)
> Each workers (in a cluster) should be configured to run a virtual monitor, since it runs as a batch.

![Train with minerl](https://tsmatz.files.wordpress.com/2020/07/20200717_training_capture.gif)

## 5. Simulate a trained agent

When you have completed a training, run and simulate your trained agent.

First, please launch Minecraft client with malmo mod.

```
cd simulate
python3 launch_client.py
```

In another shell, run and simulate your trained agent.

```
cd simulate
python3 simulate_agent.py --checkpoint_file {your trained checkpoint file}
```

You can also run a pre-trained agent in this GitHub repository (```checkpoint/checkpoint-613```).

```
cd simulate
python3 simulate_agent.py
```

![Simulate a trained agent](https://tsmatz.files.wordpress.com/2020/07/20200717_rollout_capture.gif)
