import json, os
from collections import Counter

# finding file names
def search(dirname, file_type):
    files = []
    filedirs = os.listdir(dirname)
    for file in filedirs:
        if f".{file_type}" in file:
            files.append(file)
    files.sort()  # 파일 이름대로 정렬
    return files

files = search(".", "json")
voting = []
voting_keys = []

with open (files[0], "r") as f:
    voting_keys = list(json.load(f).keys())

for file in files:
    with open (file, "r") as f:
        temp = json.load(f)
        temp = list(temp.values())
        voting.append(temp)

answers = {}
for k, v in enumerate(zip(*voting)):
    answers[voting_keys[k]] = Counter(list(v)).most_common(1)[0][0]

with open ("voting.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, indent="\t")