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


## Installation
> TODO: Release our customized OmniGibson environment setup and exact dependency versions.

This benchmark is built on a **customized OmniGibson environment**.


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
