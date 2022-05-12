import json, csv, os
from collections import Counter

# finding csv & json files
def search(dirname):
    files = []
    filedirs = os.listdir(dirname)
    for doc in filedirs:
        if doc.split(".")[-1] in ["csv", "json"]:
            files.append(doc)
    files.sort()
    return files

# finding prediction ids
def finding_keys(doc):
    voting_keys = []
    
    if doc.split(".")[-1] == "json":
        with open (doc, "r") as f:
            voting_keys = list(json.load(f).keys())
    
    else:
        with open(doc, "r") as f:
            lines = f.readlines()
            lines = list(map(lambda s: s.strip(), lines))
            lines = lines[1:-1]
            for line in lines:
                ids, texts = line.split(":")
                ids = ids.split("\"")[1]
                voting_keys.append(ids)
    
    return voting_keys

# finding prediction values
def finding_values(files):
    voting = []
    for doc in files:
        if doc.split(".")[-1] == "json":
            with open (doc, "r") as f:
                temp = json.load(f)
                temp = list(temp.values())
                voting.append(temp)
        
        else:
            temp = []
            with open(doc, "r") as f:
                lines = f.readlines()
                lines = list(map(lambda s: s.strip(), lines))
                lines = lines[1:-1]
                for line in lines:
                    ids, texts = line.split(":")
                    texts = texts.split("\"")
                    texts = "".join(texts[1:-1])
                    temp.append(texts)
            voting.append(temp)
    return voting



##### voting #####
files = search(".")  # 현재 위치에서 파일 탐색 / 제목 오름차순으로 정렬한 리스트 반환
voting_keys = finding_keys(files[0])
voting = finding_values(files)

answers = {}
for k, v in enumerate(zip(*voting)):
    answers[voting_keys[k]] = Counter(list(v)).most_common(1)[0][0]

with open ("voting.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, indent="\t")