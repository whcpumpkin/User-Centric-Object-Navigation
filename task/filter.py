import os
import json
from copy import deepcopy

def del_suffix(obj_name):

    model_name = set()

    with open('obj.json', 'r') as f:
        data = json.load(f)

    for item in data.items():
        for i in item[1]:
            model_name.add(i)
    
    obj = obj_name.split('_')
    obj = [item for item in obj if item not in model_name and not item.isdigit()]
    obj = '_'.join(obj)
 
    return obj
    
def main():

    task_file_name = 'task_to_delete.json'
    total_file = 11
    with open('scene_obj.json', 'r') as f:
        all_obj = json.load(f)
    path = '/home/user/zjy/Prompt_Navigation/task'

    data = dict()
    for scene in all_obj.keys():
        data[scene] = []

    with open(task_file_name, 'w') as f:
        json.dump(data,f,indent=4)
    
    # 每次根据merge文件数量在这里修改
    for i in range(1,total_file+1):
            
        with open(task_file_name, 'r') as f:
            data = json.load(f)
                
        origin_task_path = os.path.join(path,f'origin/whc_merge_{i}.json')
        with open(origin_task_path, 'r') as f:
            origin_task = json.load(f)

        for scene, tasks in list(origin_task.items()):

            for task in tasks:
                try:
                    task = json.loads(task)
                except:
                    task = task
                try:
                    location_list = task['location']
                except:
                    location_list = task['locations']

                new_task = deepcopy(task)
                new_task['location'] = []

                init_obj_lst = set()
                for location in location_list:
                    try:
                        location.strip('\'').strip('\"')
                        init_obj, rest = location.split('.')
                        init_obj_lst.add(init_obj)
                    except:
                        continue
                if len(init_obj_lst) != 1:
                    continue
                
                t = ['set up', 'prepare', 'Prepare']
                flag = False
                for x in t:
                    if x in new_task['user_instruction']:
                        flag = True
                        break
                if flag:
                    continue
                
                other_obj = ['chair', 'table', 'carpet', 'light', 'walls', 'floors', 'door', 'car', 'ceilings', 'window', 'cabinet', 'mirror', 'fridge', 'tent', 'washer', 'sink', 'bed']
                flag = False
                for obj in other_obj:
                    if obj in new_task['basic_object_name']:
                        flag = True
                        break
                if flag:
                    continue

                new_task['basic_object_name'] = init_obj
                if init_obj not in all_obj[scene]:
                    continue

                for location in location_list:
                    try:
                        location.strip('\'').strip('\"')
                        init_obj, rest = location.split('.')
                        action, ref_obj = rest.split('(')
                        ref_obj = ref_obj.replace(')', '')
                    except:
                        continue

                    action = action.replace('next_to', 'nextto').replace('on_top', 'ontop')
                    if action not in ['place_nextto','place_ontop','place_inside','place_under']:
                        continue
                    
                    init_obj = del_suffix(init_obj)
                    ref_obj = del_suffix(ref_obj)

                    if ref_obj == 'window' or ref_obj == 'walls':
                        continue
                    if init_obj not in all_obj[scene] or ref_obj not in all_obj[scene]:
                        continue



                    new_task['basic_object_name'] = init_obj
                    new_task['location'].append(f'{init_obj}.{action}({ref_obj})')
                
                new_task['location'] = list(set(new_task['location']))
                
                if len(new_task['location']) > 1:
                    data[scene].append(new_task)

        with open(task_file_name,'w') as f:
            json.dump(data,f,indent=4)

def count():
    with open('task.json','r') as f:
        data = json.load(f)
    
    task_cnt = 0
    location_cnt = 0

    for scene, tasks in data.items():
        task_in_scene = 0
        scene_obj=set()
        for task in tasks:
            task_cnt += 1
            task_in_scene += 1
            location_cnt += len(task['location'])
            scene_obj.add(task['basic_object_name'])
        print(f'{scene}:{task_in_scene},{len(scene_obj)}')
    
    print(f'task_cnt:{task_cnt}')
    print(f'location_pre_task:{location_cnt/task_cnt}')
    print(f'location:{location_cnt}')
   
if __name__ == '__main__':
    main()
    # count()
    