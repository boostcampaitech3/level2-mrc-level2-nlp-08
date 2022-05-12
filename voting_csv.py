import json, csv, os
from collections import Counter

# finding file names
def search(dirname, file_type):
    files = []
    filedirs = os.listdir(dirname)
    for file in filedirs:
        if f".{file_type}" in file:
            files.append(file)
    return files

files = search(".", "csv")
voting = []
voting_keys = []

with open(files[0], "r") as f:
    lines = f.readlines()
    lines = list(map(lambda s: s.strip(), lines))
    lines = lines[1:-1]
    for line in lines:
        ids, texts = line.split(":")
        ids = ids.split("\"")[1]
        voting_keys.append(ids)

for file in files:
    temp = []
    with open(file, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda s: s.strip(), lines))
        lines = lines[1:-1]
        for line in lines:
            ids, texts = line.split(":")
            texts = texts.split("\"")
            texts = "".join(texts[1:-1])
            temp.append(texts)
    voting.append(temp)

answers = {}
for k, v in enumerate(zip(*voting)):
    answers[voting_keys[k]] = Counter(list(v)).most_common(1)[0][0]

with open ("voting.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, indent="\t")