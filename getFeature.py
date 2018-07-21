from gensim.models import word2vec
import numpy as np
import Levenshtein
# from simhash import Simhash
import jieba.analyse
list_real = ['n', 'a', 'm', 'q', 'r', 'v']
lose_weight = 0
list_pass = ['o', ]
dict_real = {'n':59.5, 'a':35.5, 'm':4.5, 'q':0.5, 'r':8, 'v':24}
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
    train = []
    for item in QApairs_fenci:
        temp = calaDistance(item[0], item[1], model)
        train.append(temp)
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


def getJaroDistance(QApairs):
    temp = []
    for q, a in QApairs:
        temp.append([Levenshtein.jaro(q, a)])
    return temp


# def getSimHashDistance(QAparis):
#     temp = []
#     for q, a in QAparis:
#         temp.append(Simhash(q).distance(Simhash(a)))
#     return temp


def getUniGram(QApairs, model_name):
    temp = []
    for i in range(len(QApairs)):
        same = 0
        q_temp = jieba.analyse.extract_tags(QApairs[i][0], topK=20, withWeight=False, allowPOS=())
        q_list = [x for x in q_temp]
        a_temp = jieba.analyse.extract_tags(QApairs[i][1], topK=20, withWeight=False, allowPOS=())
        a_list = [x for x in a_temp]
        for q in q_list:
            for a in a_list:
                if q == a:
                    same+=1
        temp.append(same)
    return temp


def getWordUnigram(QApairs_fenci):
    temp = []
    for QA_fen in QApairs_fenci:
        same = 0
        for q, f in QA_fen[0]:
            for a, fa in QA_fen[1]:
                if q == a:
                    same += 1
        same2 = 0
        for i in range(len(QA_fen[0])-1):
            for j in range(len(QA_fen[1])-1):
                if QA_fen[0][i][0] ==QA_fen[1][j][0] and QA_fen[0][i+1][0] == QA_fen[1][j+1][0]:
                    same2 += 2

        temp.append(pow(same*same2, 0.5))
    return temp