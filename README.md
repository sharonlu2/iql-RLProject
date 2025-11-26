# Offline Reinforcement Learning with Implicit Q-Learning

This repository contains the official implementation of [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) by [Ilya Kostrikov](https://kostrikov.xyz), [Ashvin Nair](https://ashvin.me/), and [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).

If you use this code for your research, please consider citing the paper:
```
@article{kostrikov2021iql,
    title={Offline Reinforcement Learning with Implicit Q-Learning},
    author={Ilya Kostrikov and Ashvin Nair and Sergey Levine},
    year={2021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## How to run the code

### Install dependencies

The datasets are hosted via [Minari](https://github.com/Farama-Foundation/Minari). Ensure you can download datasets from Hugging Face (see the Minari README for authentication details if needed).

```bash
pip install --upgrade pip

pip install -r requirements.txt

# Pin protobuf for tensorboardX compatibility.
pip install "protobuf<=3.20.3"

# Installs the wheel compatible with Cuda 11 and cudnn 8.
pip install "jax[cuda111]<=0.21.1" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Also, see other configurations for CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Run training

Locomotion
```bash
python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py
```

AntMaze
```bash
python train_offline.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000
```

Kitchen and Adroit
```bash
python train_offline.py --env_name=pen-human-v0 --config=configs/kitchen_config.py
```

Finetuning on AntMaze tasks
```bash
python train_finetune.py --env_name=antmaze-large-play-v0 --config=configs/antmaze_finetune_config.py --eval_episodes=100 --eval_interval=100000 --replay_buffer_size 2000000
```

### Run on Google Colab

1. **Start a GPU runtime** in Colab (Runtime → Change runtime type → Hardware accelerator → GPU).
2. **Clone the branch you want to test** (replace `<BRANCH>` and `<FORK_OWNER>` as needed):
   ```bash
   !git clone -b <BRANCH> https://github.com/<FORK_OWNER>/iql-RLProject.git
   %cd iql-RLProject
   ```
3. **Install system and Python dependencies** (Colab supports `sudo`):
   ```bash
   !sudo apt-get update && sudo apt-get install -y patchelf
   !pip install --upgrade pip
   !pip install -r requirements.txt
   # Match CUDA on Colab (usually 11.x) for JAX; adjust the CUDA tag if Google updates the runtime.
   !pip install "jax[cuda111]<=0.21.1" -f https://storage.googleapis.com/jax-releases/jax_releases.html
   ```
4. **(If required) authenticate with Hugging Face to download Minari datasets**:
   ```bash
   !huggingface-cli login --token <YOUR_HF_TOKEN>
   ```
5. **Run training/evaluation commands** exactly as shown above (for example):
   ```bash
   !python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py
   ```
6. (Optional) **Persist outputs** by mounting Google Drive before running commands:
   ```bash
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Misc
The implementation is based on [JAXRL](https://github.com/ikostrikov/jaxrl).
