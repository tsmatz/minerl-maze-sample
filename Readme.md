# Reinforcement Learning for Minecraft (MineRL) Sample

This sample code trains an agent in Minecraft with reinforcement learning.    
In this example, an agent will learn to reach to a goal block in a given maze.

**Do not use low-spec machine**, since the training worker will request enough resources.    
(Here I used Ubuntu 18.04 on Standard D3 v2 with 4 cores and 14 GB RAM in Microsoft Azure. GPU-utilized instance will help improve batch to train more faster.)

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

MineRL requires a monitor (screen), such as VNC, to run Minecraft.    
Then, install and start service for X remote desktop.

```
sudo apt-get update
sudo apt-get install lxde -y
sudo apt-get install xrdp -y
/etc/init.d/xrdp start  # password is required
```

> Use a virtual monitor (such as, ```xvfb```) instead, when you run training as a batch in background. (Here we use a real monitor for debugging.)

Allow inbound port 3389 (remote desktop protocol) for this computer in network setting.

> You can also use SSH tunnel (port 22) instead.

Restart your computer.

Finally, install Java runtime and set ```JAVA_HOME```, since it runs on Minecraft java edition (with mods).

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

## 4. Train an agent (Reinforcement Learning)

Login to computer with remote desktop client (please use a monitor for debugging).<br>
Here I have used built-in remote desktop client in Windows, but please use appropriate X terminal client depending on your working environment. (See "Trouble Shooting" in the appendix below.)

After logging-in, clone this repository.

```
git clone https://github.com/tsmatz/minerl-maze-sample.git
cd minerl-maze-sample
```

Now run the training script (train_minerl.py) as follows.

When it launches (opens) 2 instances of Minecraft client, please close one which is not used for training. (In order to see settings in an environment, such as, action space or observation space, Ray framework will create a dummy environment inside.)

```
cd train
python3 train_minerl.py
```

> Please change ```MsPerTick``` in mission file to speed up training. (This uses 50 millisecs between ticks, which is the default value in normal Minecraft game.)<br>
> You can also run training on multiple workers in Ray cluster to speed up training. Each workers in a cluster should be configured to use a virtual monitor, since it runs as a batch in backgroud. (Using [Azure Machine Learning](https://tsmatz.wordpress.com/2018/11/20/azure-machine-learning-services/), you can quickly configure cluster with built-in RL estimator.)

## 5. Run and check a trained agent

When you have completed, run and check your trained agent.<br>
To do this,

First, login to computer with remote desktop client and launch Minecraft client with malmo mod as follows.

```
cd simulate
python3 launch_client.py
```

In another shell, run and simulate a trained agent as follows.<br>
This will run a pre-trained agent on ```checkpoint/checkpoint-645``` in this repo.

```
cd simulate
python3 simulate_agent.py
```

![Simulate a trained agent](https://tsmatz.files.wordpress.com/2020/07/20200717_rollout_capture.gif)

If you have your own trained agent, you can also run and simulate with your own checkpoint file as follows.

```
cd simulate
python3 simulate_agent.py --checkpoint_file {your trained checkpoint file}
```

## Appendix : Trouble Shooting

**Xrdp won't accept a special character for password.**

Please create a new user with a simple password.

**Error in desktop session start**

See error details in ```~/.xsession-errors```, when it has some toubles to start xrdp (X remote desktop) session. Set ```mate-session``` in ```~/.xsession``` to fix, if needed.

**Azure DSVM or ML compute**

When you use data science virtual machine (DSVM) or [AML](https://tsmatz.wordpress.com/2018/11/20/azure-machine-learning-services/) compute in Azure :

- Deactivate conda environment, since MineRL cannot be installed with conda.

```
echo -e "conda deactivate" >> ~/.bashrc
source ~/.bashrc
```

- It will include NVidia cuda, even when you run on CPU VM. This will cause a driver error ("no OpenGL context found in the current thread") when you run Minecraft java server with malmo mod.<br>
  Thereby, please ensure to uninstall cuda.

```
sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```

**Errors for display setting (monitor)**

When your application cannot detect your display (monitor), please ensure to set ```DISPLAY``` as follows.<br>
(The error message "MineRL could not detect a X Server, Monitor, or Virtual Monitor" will show up.)

```
# check your display id
ps -aux | grep vnc
# set display id (when your display id is 10)
export DISPLAY=:10
```

When you cannot directly show outputs in your physical monitor, please divert outputs through a virtual monitor (xvfb).<br>
For instance, the following will show outputs (Minecraft game) on your own VNC viewer window through a virtual monitor (xvfb).

```
# install components
sudo apt-get install xvfb
sudo apt-get install x11vnc
sudo apt-get install xtightvncviewer
# generate xvfb monitor (99) and bypass to real monitor (10)
/usr/bin/Xvfb :99 -screen 0 768x1024x24 &
/usr/bin/x11vnc -rfbport 5902 -forever -display :99 &
DISPLAY=:10 /usr/bin/vncviewer localhost:5902 &
# run program
export DISPLAY=:99
python3 train.py
```
