import json  
import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np

with open('task.json','r') as f:
    data = json.load(f)

scene_obj = dict()
for scene in data:
    scene_obj[scene] = set()
    for task in data[scene]:
        scene_obj[scene].add(task['basic_object_name'])
    print(scene,len(scene_obj[scene]))


# plt.bar(scenes, lens)
# plt.xticks(scenes, rotation=90)
# plt.show()

