import json

with open('obj.json','r') as f:
    data = json.load(f)

category = list(data.keys())

id2sem, sem2id = dict(), dict()
for i in range(len(category)):
    id2sem[i] = category[i]
    sem2id[category[i]] = i
with open('sem2id.json', 'w') as f:
    json.dump(sem2id,f)
with open('id2sem.json', 'w') as f:
    json.dump(id2sem,f)