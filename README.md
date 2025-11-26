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

# Remove any preinstalled JAX packages (Colab often ships newer nightly builds
# that are incompatible with this repository's pinned stack).
pip uninstall -y jax jaxlib

# Install a JAX/JAXLIB pair that has wheels for Python 3.12 and matches modern
# Flax/Optax releases.
pip install "jax==0.4.38"
pip install "jaxlib==0.4.38"  # CPU-only; see JAX release notes for CUDA wheels
```

If you see errors such as `ImportError: cannot import name 'linear_util' from 'jax'` or warnings about `jax_cuda12_plugin` being
ignored, it means a conflicting JAX/JAXLIB build was preinstalled. Re-running the uninstall/install commands above ensures Flax
and JAX are aligned on a compatible, wheel-available version.

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
   # Match CUDA on Colab (usually 11.x) for JAX. CUDA wheels for jaxlib 0.4.38
   # are available for common CUDA versions on Python 3.10–3.12; if pip cannot
   # find a matching CUDA wheel, fall back to the CPU wheel shown here.
   !pip uninstall -y jax jaxlib
   !pip install "jax==0.4.38"
   !pip install "jaxlib==0.4.38"  # CPU-only; see JAX release notes for CUDA wheels
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
