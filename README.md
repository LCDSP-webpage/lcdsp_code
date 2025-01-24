# Language-Controlled Diverse Style Policies

Welcome to the official PyTorch implementation of the paper:

"Complex Instruction Following with Diverse Style Policies" 

This repository includes the code and model for grf environment inference. The training code is currently being refactored and organized for better readability and reproducibility.

Here is an overview of the inference process:

<img src="figures\inference_process.png" alt="LCDSP_inference" width="800">

Here is an overview of the DTP prompt:

<img src="figures\prompt.png" alt="LCDSP_prompt" width="800">

## Environment

Check details in Google Research Football [football](https://github.com/google-research/football/tree/9a9e35bcd1929a82c2b91eeed777e5e571f29d38)

## GRF Installation

Full details of the installation can be found in the [Compiling Google Research Football Engine](https://github.com/google-research/football/blob/master/gfootball/doc/compile_engine.md#windows), we provide installation guide for windows platforms here.

### 1. Create conda environment

Create a new conda environment and install the required packages.

```
conda create -n grf python=3.8
conda activate grf
# depend on your cuda versiopn
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Other packages can be installed using the following command.

```python
pip install --upgrade pip setuptools wheel
pip install psutil
pip install gym==0.16.0
```

### 2. Install dependency

- [Git](https://git-scm.com/downloads/win)

- [Visual Studio 2019 Community Edition](https://visualstudio.microsoft.com/zh-hans/downloads/) (make sure to select "Desktop development with C++" component)

- [CMake](https://cmake.org/download/)

- [vcpkg](https://github.com/microsoft/vcpkg)

First, install Git and VS2019. Then, follow the steps below to install CMake and vcpkg.

CMake is installed using the MSI installer, with the option to add CMake to the environment variables automatically selected during installation. After installation, verify whether CMake is included in the environment variables.

>cmake --version

If CMake does not appear, but "C:\Program Files\CMake\bin" is already listed in the "Path" variable under "System variables" in the environment variables window, a system restart is required.

If you prefer not to restart, you can use temporary environment variables. Open a new CMD window and execute the command, ensuring not to use Bash or PowerShell, as the path format differs.

```
set PATH=C:\Program Files\CMake\bin;%PATH%
cmake --version
```

Install vcpkg in the same CMD window; otherwise, you'll need to set the temporary environment variables again.

```
cd C:\dev
git clone https://github.com/microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
```

### 3. Install GRF in conda environment

```python
python -m pip install gfootball
```

### 4. Copy custom scenarios to conda environment

To use our custom-configured scenario, it needs to be copied into the scenario folder of the gfootball package within the conda grf environment. Run the script using a bash command, or manually perform the copy operation.

```bash 
cp -r env/random_ball_scenarios/* your_conda_path/envs/grf/Lib/site-packages/gfootball/scenarios
```

## Installation of dependencies related to inference code

```bash
pip install -r requirements.txt
```

## Run a game

After starting the game, make sure to click elsewhere with the mouse, and avoid clicking on the game screen afterward to prevent it from freezing.

Use Ctrl+C in the command line to close the game and the input subprocess.

---
### Choose a scenario, run with base style parameters

The following command will run using a pre-configured style parameter, and in the 5v5 mode, the opponent will also use this style parameter.

The team wearing yellow uniforms is my_ai, the team wearing blue uniforms is the opponent.

#### Sigle-player scenario

```python
python run_log.py --my_ai lcdsp_single_player --opponent noop_AI --env instruction_follow_single --agent_type agents_single_player
```

---
#### Two-player scenario

```python
python run_log.py --my_ai lcdsp_two_player --opponent noop_AI --env instruction_follow_2v2 --agent_type agents_two_player
```

---
#### 5v5 scenario

The opponent can choose to engage in self-play using lcdsp_two_player or compete against noop_AI and buildin_AI.

```python
python run_log.py --my_ai lcdsp_5v5 --opponent lcdsp_5v5 --env football_5v5_malib --agent_type agents_5v5
```

### Contorl policy with style-parameter

<video width="1000" controls>
  <source src="figures\param_control.mp4" type="video/mp4">
</video>

In all three scenarios, the corresponding style parameters can be adjusted through Box UI input parameters. You only need to set style_input to True in the arguments. An input box will appear shortly thereafter. 

After launching the game environment, enter the desired style parameters and simply click submit, you can change the style parameters while the environment is running. You can also modify the initial style parameters in the corresponding JSON file within the *base_style_parameters* folder before launching.

The team wearing yellow uniforms is the instruction-controlled team, attacking towards the right side of the field. 

#### Sigle-player scenario

```python
python run_log.py --my_ai lcdsp_single_player --opponent noop_AI --env instruction_follow_single --agent_type agents_single_player --style_input True
```

Below are the adjustable style parameters for a single-player scenario and their corresponding meanings:

- active_area_x
    - `0`  front
    - `1` middle
    - `2` back
- active_area_y
    - `0` left
    - `1` center
    - `2` right
- shot 
    - `0` goal area shot
    - `1` penalty area shot
- move
    - `0` run
    - `1` sprint
    - `2` dribble

The remaining style parameters not mentioned for this scenario are not utilized and should retain the same values as the default configuration.

To execute the *shooting* behavioral style, set the active_area to 0 and goal to 0.1. Set *move* and *shot* as one-hot vectors to select the corresponding behaviors.

To execute the *navigate* behavioral style, set the *shot* to 0 and *goal* to 0. Set *move* and *active_area_x*, *active_area_y* as one-hot vectors to select the corresponding actions.

---
#### Two-player scenario

```python
python run_log.py --my_ai lcdsp_two_player --opponent noop_AI --env instruction_follow_2v2 --agent_type agents_two_player --style_input True
```

Below are the adjustable style parameters for a two-player scenario and their corresponding meanings:

- formation
    - `0`  front
    - `1` middle
    - `2` back
- move
    - `0` run
    - `1` sprint
    - `2` dribble
- shot 
    - `0` goal area shot
    - `1` penalty area shot
- hold_ball
    - `0` not active
    - `1` active
- pass
    - `0` not active
    - `1` active

The remaining style parameters not mentioned for this scenario are not utilized and should retain the same values as the default configuration.

- To execute the *shooting* behavioral style, set *goal* to 0.5, and set *pass* and *hold_ball* to 0. Combining the indices of *shot* and *move* can generate corresponding behaviors.

- To execute the pass behavioral style, set *pass* to 1, and set *shot*, *goal*, and *hold_ball* to 0. Adjusting *formation* allows for changes in the formation.

- To execute the hold ball behavioral style, Set hold_ball to 1, and set *shot*, *goal*, and *pass* to 0. Adjusting *formatio*n allows for changes in the formation.



---
#### 5v5 scenario

The opponent can choose to engage in self-play using lcdsp_5v5 or compete against noop_AI and buildin_AI.

```python
python run_log.py --my_ai lcdsp_5v5 --opponent lcdsp_5v5 --env football_5v5_malib --agent_type agents_5v5 --style_input True
```

- In the 5v5 environment, all parameters can be freely combined, allowing you to configure them according to your preferences to create the desired style.

### Contorl policy with natural-language instructions

<video width="1000" controls>
  <source src="figures\language_control.mp4" type="video/mp4">
</video>

In all three scenarios, natural language input can be used to direct the multi-style policy to follow instructions, it requires using your own OpenAI API Key at language_control/llm.py *completion* function. 

You can select the instructions used in the paper from language_control/scenario/instructions, or input your own desired instruction.

If a command cannot be parsed into style parameters, it will revert to the default style.

#### Sigle-player scenario

```python
python run_log.py --my_ai lcdsp_single_player --opponent noop_AI --env instruction_follow_single --agent_type agents_single_player --language_input True
```

---
#### Two-player scenario

```python
python run_log.py --my_ai lcdsp_two_player --opponent noop_AI --env instruction_follow_2v2 --agent_type agents_two_player --language_input True
```

---
#### 5v5 scenario

The opponent can choose to engage in self-play using lcdsp_two_player or compete against noop_AI and buildin_AI.

```python
python run_log.py --my_ai lcdsp_5v5 --opponent lcdsp_5v5 --env football_5v5_malib --agent_type agents_5v5 --language_input True
```

The following diagram illustrates the behaviors of the Counter Attack and Tiki-Taka tactics.

<img src="figures\two_tactics.png" alt="5v5 tactics" width="1000">


## Train your own policy
The training code is currently being refactored and organized for better readability and reproducibility.