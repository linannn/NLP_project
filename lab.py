import prepare
import getFeature
import numpy as np
import train
import sys

wordModel = 'data/train.model'

# 分词与生成word2vec
def train_fenci(trainFile):
    print("train file fenci")
    prepare.save_fenci(trainFile, trainFile+'.fen')
    prepare.getWord2Vec(wordModel)


def test_fenci(testFile):
    print("test file fenci")
    prepare.save_fenci(testFile, testFile+'.fen')


def store_data(fileName):
    print("calculate and store data "+fileName)
    QAparis_fenci = prepare.load_fenci(fileName+'.fen')
    QAparis, tags = prepare.load_QApairs(fileName)
    cos_dis = getFeature.getCosDistance(QAparis_fenci, wordModel)
    word_dis = getFeature.getWordLengthDiff(QAparis)
    phase_dis = getFeature.getWordLengthDiff(QAparis_fenci)
    edit_dis = getFeature.getEditDistance(QAparis)
    jaro_temp = getFeature.getJaroDistance(QAparis)
    jaro_dis = [x[0] for x in jaro_temp]
    unigram = getFeature.getUniGram(QAparis, wordModel)
    word_uni = getFeature.getWordUnigram(QAparis_fenci)
    prepare.save_feature_data(cos_dis, fileName+'.cos')
    prepare.save_feature_data(word_dis, fileName+'.word')
    prepare.save_feature_data(phase_dis, fileName+'.phase')
    prepare.save_feature_data(edit_dis, fileName+'.edit')
    prepare.save_feature_data(jaro_dis, fileName+'.jaro')
    prepare.save_feature_data(unigram, fileName+'.unigram')
    prepare.save_feature_data(word_uni, fileName+'.worduni')


def load_data(fileName):
    print("load data "+ fileName)
    cos = prepare.load_feature_data(fileName+'.cos')
    word = prepare.load_feature_data(fileName+'.word')
    phase = prepare.load_feature_data(fileName+'.phase')
    edit = prepare.load_feature_data(fileName+'.edit')
    jaro = prepare.load_feature_data(fileName+'.jaro')
    unigram = prepare.load_feature_data(fileName+'.unigram')
    worduni = prepare.load_feature_data(fileName + '.worduni')
    return np.column_stack((cos, word, phase, edit, jaro, unigram, worduni))


if __name__ == '__main__':
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    scoreFile = sys.argv[3]
    newTrain = int(sys.argv[4])
    if(newTrain == 1):
        train_fenci(trainFile)
        store_data(trainFile)
    test_fenci(testFile)
    store_data(testFile)
    QA, tags = prepare.load_QApairs(trainFile)
    print("train start")
    train.trainByXgboost(load_data(trainFile), tags, load_data(testFile), scoreFile)
