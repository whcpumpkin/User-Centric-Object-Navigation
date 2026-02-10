import json 

with open('task.json', 'r') as f:
    data = json.load(f)

objs = set()
for tasks in data.values():
    for task in tasks:
        locations = task['location']
        for location in locations:
            obj1, t = location.split('.')
            objs.add(obj1)
            _, obj2 = t.split('(')
            obj2 = obj2[:-1]
            objs.add(obj2)
objs = list(objs)
objs.sort()
data = {}
for idx, obj in enumerate(objs,start=0):
    data[str(idx)] = obj
with open('id2sem.json', 'w') as f:
    json.dump(data,f)
data = {}
for idx, obj in enumerate(objs,start=0):
    data[obj] = str(idx)
with open('sem2id.json', 'w') as f:
    json.dump(data,f)
