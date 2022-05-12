from tqdm import tqdm
import pickle
import datasets.arrow_dataset as da
from datasets import load_dataset
import pandas as pd

def erase_special(dataset):
    for i in tqdm(range(len(dataset))):
        if dataset['answers'][i]['text'][0].endswith(')'):
            dataset['answers'][i]['text'][0] = dataset['answers'][i]['text'][0].split("(")[0]


def use_jaccarad_similarity(dataset):
    train_data = dataset.to_pandas()

    def get_jaccard_sim(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    sim = []
    for i in tqdm(range(len(train_data))):
        sim.append(get_jaccard_sim(train_data['question'][i], train_data['title'][i]))

    train_data['sim'] = sim

    train_data.sort_values('sim', ascending=False, inplace=True)

    del train_data['sim'], train_data['__index_level_0__']

    train_dataset = da.Dataset.from_pandas(train_data)

    return train_dataset

def concat_koquad(train_dataset):
    ###############
    # Kosquad 합치는 방법
    kosquad = load_dataset("squad_kor_v1")

    df = train_dataset.to_pandas()
    df2 = kosquad['train'].to_pandas()[['title', 'context', 'question', 'answers']]
    # df3 = kosquad['validation'].to_pandas()[['title', 'context', 'question', 'answers']]

    df = pd.concat([df, df2], axis=0).sample(frac=1).reset_index(drop=True)

    train_dataset = da.Dataset.from_pandas(df)

    return train_dataset

def change_question(datasets, mode = 'valid'):
    check = ['은?', '는?', '이?']
    data = datasets.to_pandas()

    if mode=='train':
        with open("pickle/a.pickle", "rb") as f:
            ans = pickle.load(f)  # 개발자가 만든 Class의 Instance 또한 저장 가능

        with open("pickle/b.pickle", "rb") as f:
            ori = pickle.load(f)  # 개발자가 만든 Class의 Instance 또한 저장 가능
    elif mode=='test':
        with open("pickle/f.pickle", "rb") as f:
            ori = pickle.load(f)  # 개발자가 만든 Class의 Instance 또한 저장 가능

        with open("pickle/e.pickle", "rb") as f:
            ans = pickle.load(f)  # 개발자가 만든 Class의 Instance 또한 저장 가능
    else:
        with open("pickle/d.pickle", "rb") as f:
            ans = pickle.load(f)  # 개발자가 만든 Class의 Instance 또한 저장 가능

        with open("pickle/c.pickle", "rb") as f:
            ori = pickle.load(f)  # 개발자가 만든 Class의 Instance 또한 저장 가능

    for i in tqdm(range(len(ans))):
        string = ori[i].split()[-1]
        for j in range(3):
            if string.endswith(check[j]):
                if 'who' in ans[i] or "사람" in ori[i]:
                    data['question'][i] = data['question'][i].replace("?", " 누구일까요?")
                elif 'where' in ans[i] or "장소" in ori[i]:
                    data['question'][i] = data['question'][i].replace("?", " 어디일까요?")
                elif "when" in ans[i] or "언제" in ori[i]:
                    data['question'][i] = data['question'][i].replace("?", " 언제일까요?")
                else:
                    data['question'][i] = data['question'][i].replace("?", " 무엇일까요?")
                break
            elif '?' not in string:
                if 'who' in ans[i] or "사람" in ori[i]:
                    data['question'][i] = data['question'][i] + " 누구일까요?"
                elif 'where' in ans[i] or "장소" in ori[i]:
                    data['question'][i] = data['question'][i] + " 어디일까요?"
                elif "when" in ans[i] or "언제" in ori[i]:
                    data['question'][i] = data['question'][i] + " 언제일까요?"
                else:
                    data['question'][i] = data['question'][i] + " 무엇일까요?"

                break

    train_dataset = da.Dataset.from_pandas(data)

    return train_dataset