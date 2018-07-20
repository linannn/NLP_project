import jieba
import jieba.posseg
from gensim.models import word2vec
def save_fenci(inputFile, fenciFile):
    f = open(inputFile, encoding='UTF-8')
    fw = open(fenciFile, 'w', encoding='UTF-8')
    ff = open('data/temp', 'w', encoding='UTF-8')
    for line in f:
        [ques, ans, rela] = line.strip('﻿').strip('\n').split('\t')
        ques_flag = [(x, y) for x, y in jieba.posseg.cut(ques)]
        ans_flag = [(x, y) for x, y in jieba.posseg.cut(ans)]
        for w, fl in ques_flag:
            fw.write(w+'$$$$'+fl+'&&&&')
            ff.write(w+' ')
        fw.write('\t')
        for w, fl in ans_flag:
            fw.write(w+'$$$$'+fl+'&&&&')
            ff.write(w+' ')
        fw.write('\t')
        fw.write(rela+'\n')
    f.close()
    fw.close()


def load_fenci(inputFile):
    f = open(inputFile, encoding='UTF-8')
    qa_fenci = []
    for line in f:
        [q, a, rela] = line.strip('﻿').strip('\n').split('\t')
        q = q.strip('&&&&')
        qtags = q.split('&&&&')
        question_flag = []
        for s in qtags:
            question_flag.append(tuple(s.split('$$$$')))
        a = a.strip('&&&&')
        atags = a.split('&&&&')
        answer_flag = []
        for s in atags:
            answer_flag.append(tuple(s.split('$$$$')))
        qa_fenci.append([question_flag, answer_flag])
    return qa_fenci


def load_QApairs(inputFile):
    f = open(inputFile, encoding='UTF-8')
    pairs = []
    tags = []
    for line in f:
        [ques, ans, t] = line.strip('﻿').strip('\n').split('\t')
        pairs.append([ques, ans])
        tags.append(t)
    f.close()
    return pairs, tags


def getWord2Vec(moduleName):
    sentences = word2vec.Text8Corpus(u"data/temp")
    model = word2vec.Word2Vec(sentences, sg=1, size=150,  window=5,  min_count=1)
    model.save(moduleName)


def save_feature_data(feature, fileName):
    f = open(fileName, 'w', encoding='UTF-8')
    for i in feature:
        f.write(str(i)+'\n')


def load_feature_data(fileName):
    temp = []
    f = open(fileName, encoding='UTF-8')
    for line in f:
        temp.append(float(line))
    return temp