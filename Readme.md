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

MineRL requires a monitor (screen), such as VNC, to run Minecraft.    
Then, install and start service for X remote desktop.

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

Login to computer with remote desktop client (with a monitor).

When it launches 2 instances of Minecraft client, please close one, which is not used for training. (Ray will create a dummy env in order to see settings in environment, such as, action space or observation space.)

```
cd train
python3 train_minerl.py
```

> This training will take so long time, since it runs on a single instance. To run distributed training, configure ray cluster (multiple workers). (You can quickly configure using [Azure Machine Learning](https://tsmatz.wordpress.com/2018/11/20/azure-machine-learning-services/) on cloud.)    
> Each workers (in a cluster) should be configured to run a virtual monitor, since it runs as a batch.

## 5. Simulate a trained agent

When you have completed a training, run and check your trained agent.

Login to computer with remote desktop client (with a monitor).

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

This repository includes a pre-trained agent (```checkpoint/checkpoint-645```).    
Then you can also run this checkpoint as follows.

```
cd simulate
python3 simulate_agent.py
```

![Simulate a trained agent](https://tsmatz.files.wordpress.com/2020/07/20200717_rollout_capture.gif)

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
