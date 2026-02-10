import os
import random
import yaml
from argparse import Namespace

import omnigibson as og
from omnigibson.envs.ucon_env import UcON_Environment

TASK_JSON = os.environ.get("UCON_TASK_JSON", "/ucon/task/task_nodoor.json")

cfg_path = os.path.join(og.example_config_path, "local_explore.yaml")
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)


args = Namespace(
    step_size=0.25,
    turn_size=1.570796326794896619,
    task_path=TASK_JSON,
    target_obj_num=5,
    scene="default",
    max_step=50,
    map_collect=0,   
)

env = UcON_Environment(configs=cfg, args=args)
obs = env.reset()
print("reset ok, obs type:", type(obs))

for t in range(10):
    action = random.choice(
        ["move_forward", "turn_left", "turn_right", "look_up", "look_down", "open"]
    )
    obs, r, done, info = env.step(action)
    print(f"{t}: action={action}, reward={r}, done={done}")
    if done:
        break

env.close()
print("smoke test done")
