# UcON: User-Centric Object Navigation (ICRA 2026)

Official repository for the ICRA 2026 paper:  
**User-Centric Object Navigation: A Benchmark with Integrated User Habits for Personalized Embodied Object Search.**


## Status
This repository is being prepared for open-sourcing.
Currently, it contains the README and basic project information.
Code, task resources, and evaluation scripts will be released after cleanup and verification.


## TODOs (Under Development)
- [x] README
- [ ] Environment setup
- [ ] Task resources
- [ ] Evaluation scripts

## Benchmark
UcON evaluates an agent’s ability to leverage a **User Habit Knowledge Base (UHKB)** to locate a target object category in a scene.

### Scale & setup
- 489 target object categories; ~22,600 natural-language habits.
- Built on a customized OmniGibson-based simulator with 22 initial scenes.
- Each episode provides a large UHKB; only a small fraction is relevant to the current target.

### Task definition
In each task instance:
- The agent is initialized in a habit-shaped scene.
- The goal is to locate a target object category given a UHKB.

### Episode definition
- Action space: `MoveAhead`, `RotateLeft`, `RotateRight`, `LookUp`, `LookDown`, `Open`, `Done`
  - `MoveAhead`: 0.25m
  - `RotateLeft/RotateRight`: 90°
  - `LookUp/LookDown`: 30°
- `Open`: opens eligible containers within FoV and distance threshold `d_open = 1m`
- Success: call `Done` when the target is visible and within `d_succ = 1m`
- Max episode length: 300 steps

### Metrics
- Success Rate (SR)
- SPL (Success weighted by Path Length)


## Environment Setup

We recommend using the official OmniGibson Docker image as the base runtime, then applying UcON’s small code patches inside the container.

### Prerequisites
A machine with an NVIDIA GPU + working driver
Docker installed, plus NVIDIA Container Toolkit enabled (so --gpus all works)
This repo cloned on the host (example path below)

### 1) Pull the official OmniGibson image

```
docker pull stanfordvl/omnigibson:1.0.0
```

### 2) Start a container
On the host, set two paths:
- `UCON_DIR`: where you cloned this repo
- `OG_DATA_DIR`: a persistent directory to store OmniGibson assets/datasets (recommended)


```
# Example: edit these two lines to your own paths
export UCON_DIR=/path/to/UcON
export OG_DATA_DIR=/path/to/og_data

cd "$UCON_DIR"

docker run --gpus all -it \
  -e OMNIGIBSON_HEADLESS=1 \
  -v "$OG_DATA_DIR":/data/og_data \
  -v "$UCON_DIR":/ucon \
  --name ucon-dev \
  stanfordvl/omnigibson:1.0.0 \
  /bin/bash
```
#### Notes
- `/data/og_data` is the in-container path; we mount `OG_DATA_DIR` from the host so datasets persist across container recreation.
- `UCON_DIR` is mounted to `/ucon` for easy patching and running scripts.


### 3) Apply UcON patches
In the container:
```
cd /ucon
bash sync_ucon.sh
```

This syncs the modified OmniGibson files shipped with UcON into the installed OmniGibson source tree used by the container.

### 4) Download required datasets
In the container:

```
cd /omnigibson-src
python scripts/download_datasets.py
```
Datasets will be downloaded under the mounted path /data/og_data (so they persist even if you recreate the container).

### 5) Run a smoke test
In the container:
```
cd /ucon
bash scripts/smoke_test.sh
```

## Task resources
> TODO: Release the task resources and generation scripts and document the format.

UcON releases base task resources and generates concrete tasks on-the-fly in a customized simulator environment.

## Evaluation
> TODO: Provide runnable commands for all baselines and reproduce the main results.

We will provide scripts to evaluate various baselines on UcON.


## Citation
Citation information (BibTeX / public link) will be provided after the camera-ready / public release.


## Contact
If you have any suggestions or questions, please feel free to contact us:

- [Hongcheng Wang](https://whcpumpkin.github.io): whc.1999@pku.edu.cn
- Jinyu Zhu: zhujinyu@stu.pku.edu.cn
- [Hao Dong](https://zsdonghao.github.io): hao.dong@pku.edu.cn