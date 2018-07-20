import prepare
import getFeature
import numpy as np
import train
develop_file = 'data/develop.data'
train_file = 'data/training.data'
develop_fenci = 'data/develop.fen'
train_fenci = 'data/train.fen'
develop_model = 'data/develop.model'
train_model = 'data/train.model'
score = 'data/score'

# 分词，生成word2vec
# prepare.save_fenci(train_file,train_fenci)
# prepare.getWord2Vec(train_model)
# prepare.save_fenci(develop_file, develop_fenci)
# prepare.getWord2Vec(develop_model)

# 训练集
# '''
print("load train data set")
QAparis_fenci = prepare.load_fenci(train_fenci)
QAparis, tags = prepare.load_QApairs(train_file)
# cos_dis = getFeature.getCosDistance(QAparis_fenci, train_model)
# prepare.save_feature_data(cos_dis, 'data/traincos')
cos_dis = prepare.load_feature_data('data/traincos')
word_length_diff = getFeature.getWordLengthDiff(QAparis)
word_phase_diff = getFeature.getWordLengthDiff(QAparis_fenci)
edit_dis = getFeature.getEditDistance(QAparis)
jaro_dis = getFeature.getJaroDistance(QAparis)
# simhash_dis = getFeature.getSimHashDistance(QAparis)
unigram = getFeature.getUniGram(QAparis, train_model)
prepare.save_feature_data(unigram, 'data/trainunigram')
unigram = prepare.load_feature_data('data/trainunigram')
train_x = np.column_stack((cos_dis, word_length_diff, word_phase_diff, edit_dis, jaro_dis,
                           unigram))
# '''
# 测试集
print("load test data set")
QAparis_fenci_predit = prepare.load_fenci(develop_fenci)
QAparis_predit, tags_predit = prepare.load_QApairs(develop_file)
cos_dis_predit = getFeature.getCosDistance(QAparis_fenci_predit, train_model)
word_length_diff_predit = getFeature.getWordLengthDiff(QAparis_predit)
word_phase_diff_predit = getFeature.getWordLengthDiff(QAparis_fenci_predit)
edit_dis_predit = getFeature.getEditDistance(QAparis_predit)
jaro_dis_predit = getFeature.getJaroDistance(QAparis_predit)
# simhash_dis_predit = getFeature.getSimHashDistance(QAparis_predit)
unigram_predit = getFeature.getUniGram(QAparis_predit, train_model)
prepare.save_feature_data(unigram_predit, 'data/developunigram')
unigram_predit = prepare.load_feature_data('data/developunigram')
test_x = np.column_stack((cos_dis_predit, word_length_diff_predit, word_phase_diff_predit, edit_dis_predit, jaro_dis_predit,
                          unigram_predit))

print("train start")
train.trainByXgboost(train_x, tags, test_x, score)