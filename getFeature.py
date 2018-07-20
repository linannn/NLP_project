from gensim.models import word2vec
import numpy as np
import Levenshtein
list_real = ['n', 'a', 'm', 'q', 'r', 'v']
lose_weight = 0
list_pass = ['o', ]
# dict_real = {'n':10, 'a':10, 'm':10, 'q':10, 'r':10, 'v':10}#59.5
dict_real = {'n':59.5, 'a':35.5, 'm':4.5, 'q':0.5, 'r':8, 'v':24}#59.5
def getLoseWeight(WFpairs):
    flags = list(reversed([f for s, f in WFpairs]))
    loss = [1] * len(flags)
    pre_flag = 'a'
    for i in range(len(flags)):
        if pre_flag == 'n' and flags[i][0] == 'n':
            loss[i] = lose_weight
        pre_flag = flags[i][0]
    return list(reversed(loss))


def cosVector(WFpairs, model):
    vector = np.zeros(150)
    loss_list = getLoseWeight(WFpairs)
    for i in range(len(WFpairs)):
        (word, flag) = WFpairs[i]
        loss = loss_list[i]
        try:
            if flag[0] in list_pass:
                continue
            elif flag[0] in list_real:
                vector = vector + model[word] * dict_real[flag[0]] * loss
            else:
                vector = vector + model[word] * loss
        except:
            continue
    return vector


def calaDistance(ques, ans, model):
    ques_vector = cosVector(ques, model)
    ans_vector = cosVector(ans, model)
    num = np.dot(ques_vector, np.transpose(ans_vector))
    distance = np.linalg.norm(ques_vector) * np.linalg.norm(ans_vector)
    return 0.5 + 0.5 * num / distance


def getCosDistance(QApairs_fenci, model_name):
    model  = word2vec.Word2Vec.load(model_name)
    fw = open('data/cosDistance', 'w', encoding='UTF-8')
    train = []
    for item in QApairs_fenci:
        temp = calaDistance(item[0], item[1], model)
        fw.write(str(temp)+'\n')
        train.append(temp)
    fw.close()
    return train


def getWordLengthDiff(QAparis):
    length_diff = [len(q) - len(a) for q, a in QAparis]
    return length_diff


def getPhraseDiff(QApairs_fenci):
    length_deff = [len(q) - len(a) for q, a in QApairs_fenci]
    return length_deff


def getEditDistance(QApairs):
    temp = []
    for q, a in QApairs:
        temp.append(Levenshtein.distance(q, a))
    return temp
