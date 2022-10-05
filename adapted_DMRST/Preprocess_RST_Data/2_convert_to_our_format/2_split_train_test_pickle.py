from glob import glob
import pickle
import numpy as np
import os

import random
random.seed(666)

en_rstdt_test_samples_list = open("../../data/en-rstdt-test-list.txt", "r", encoding="utf8").read().strip().split("\n")
en_gum_test_samples_list = open("../../data/en-gum-test-list.txt", "r", encoding="utf8").read().strip().split("\n")
zh_gum_test_samples_list = open("../../data/zh-gcdt-test-list.txt", "r", encoding="utf8").read().strip().split("\n")
bi_gum_test_samples_list = open("../../data/bi-gum-test-list.txt", "r", encoding="utf8").read().strip().split("\n")
fivebi_gum_test_samples_list = open("../../data/fivebi-gum-test-list.txt", "r", encoding="utf8").read().strip().split("\n")
fiveen_gum_test_samples_list = open("../../data/fiveen-gum-test-list.txt", "r", encoding="utf8").read().strip().split("\n")
en_rstdt_dev_samples_list = open("../../data/en-rstdt-dev-list.txt", "r", encoding="utf8").read().strip().split("\n")
en_gum_dev_samples_list = open("../../data/en-gum-dev-list.txt", "r", encoding="utf8").read().strip().split("\n")
zh_gum_dev_samples_list = open("../../data/zh-gcdt-dev-list.txt", "r", encoding="utf8").read().strip().split("\n")
bi_gum_dev_samples_list = open("../../data/bi-gum-dev-list.txt", "r", encoding="utf8").read().strip().split("\n")
fivebi_gum_dev_samples_list = open("../../data/fivebi-gum-dev-list.txt", "r", encoding="utf8").read().strip().split("\n")
fiveen_gum_dev_samples_list = open("../../data/fiveen-gum-dev-list.txt", "r", encoding="utf8").read().strip().split("\n")


def save_pickle(obj, file_path):
    file = open(file_path, 'wb')
    pickle.dump(obj, file)
    file.close()


def find_english_dev_test(FileNames, language):
    dev_indexes = []
    test_indexes = []
    leave_out_indexs = []
    if language.startswith("en-gum4news") or language.startswith("en-gum2rstdt4news"): # Janet EMNLP 2022 experiments
        test_samples_list = ["GUM_news_nasa", "GUM_news_sensitive", "GUM_news_homeopathic", "GUM_news_iodine"]
        dev_samples_list = en_gum_dev_samples_list
    elif language.startswith("en-rstdt"):
        test_samples_list = en_rstdt_test_samples_list
        dev_samples_list = en_rstdt_dev_samples_list
    elif language.startswith("en-gum"):
        test_samples_list = en_gum_test_samples_list
        dev_samples_list = en_gum_dev_samples_list
    elif language.startswith("zh-gcdt"):
        test_samples_list = zh_gum_test_samples_list
        dev_samples_list = zh_gum_dev_samples_list
    elif language.startswith("bi-gum"):
        test_samples_list = bi_gum_test_samples_list
        dev_samples_list = bi_gum_dev_samples_list
    elif language.startswith("fivebi-gum"):
        test_samples_list = fivebi_gum_test_samples_list
        dev_samples_list = fivebi_gum_dev_samples_list
    elif language.startswith("outbi2en-gum"):
        test_samples_list = en_gum_test_samples_list
        dev_samples_list = en_gum_dev_samples_list
        leave_out_indexs = zh_gum_test_samples_list
    elif language.startswith("outbi2zh-gcdt"):
        test_samples_list = zh_gum_test_samples_list
        dev_samples_list = zh_gum_dev_samples_list
        leave_out_indexs = en_gum_test_samples_list
    elif language.startswith("outfivebi2en-gum"):
        test_samples_list = fiveen_gum_test_samples_list
        dev_samples_list = fiveen_gum_dev_samples_list
        leave_out_indexs = zh_gum_test_samples_list
    elif language.startswith("outfivebi2zh-gcdt"):
        test_samples_list = zh_gum_test_samples_list
        dev_samples_list = zh_gum_dev_samples_list
        leave_out_indexs = fiveen_gum_test_samples_list
    elif language.startswith("outtwelve2fiveen-gum"):
        test_samples_list = fiveen_gum_test_samples_list
        dev_samples_list = fiveen_gum_dev_samples_list
        leave_out_indexs = en_gum_test_samples_list
    elif language.startswith("fiveen-gum"):
        test_samples_list = fiveen_gum_test_samples_list
        dev_samples_list = fiveen_gum_dev_samples_list
    elif language.startswith("fivebitransen2en-gum"):
        test_samples_list = fiveen_gum_test_samples_list
        dev_samples_list = fiveen_gum_dev_samples_list
        leave_out_indexs = zh_gum_test_samples_list
    elif language.startswith("fivebitransen2zh-gcdt"):
        test_samples_list = zh_gum_test_samples_list
        dev_samples_list = zh_gum_dev_samples_list
        leave_out_indexs = fiveen_gum_test_samples_list
    elif language.startswith("bitransen2en-gum"):
        test_samples_list = en_gum_test_samples_list
        dev_samples_list = en_gum_dev_samples_list
        leave_out_indexs = zh_gum_test_samples_list
    elif language.startswith("bitransen2zh-gcdt"):
        test_samples_list = zh_gum_test_samples_list
        dev_samples_list = zh_gum_dev_samples_list
        leave_out_indexs = en_gum_test_samples_list
    elif language.startswith("fivebitranszh2en-gum"):
        test_samples_list = fiveen_gum_test_samples_list
        dev_samples_list = fiveen_gum_dev_samples_list
        leave_out_indexs = zh_gum_test_samples_list
    elif language.startswith("fivebitranszh2zh-gcdt"):
        test_samples_list = zh_gum_test_samples_list
        dev_samples_list = zh_gum_dev_samples_list
        leave_out_indexs = fiveen_gum_test_samples_list
    elif language.startswith("bitranszh2en-gum"):
        test_samples_list = en_gum_test_samples_list
        dev_samples_list = en_gum_dev_samples_list
        leave_out_indexs = zh_gum_test_samples_list
    elif language.startswith("bitranszh2zh-gcdt"):
        test_samples_list = zh_gum_test_samples_list
        dev_samples_list = zh_gum_dev_samples_list
        leave_out_indexs = en_gum_test_samples_list
    else:
        assert False

    for i, name in enumerate(FileNames):
        if name in test_samples_list:
            test_indexes.append(i)
        elif name in dev_samples_list:
            dev_indexes.append(i)

    if language.startswith("en-gum4news") or language.startswith("en-gum2rstdt4news"): # Janet EMNLP 2022 experiments
        assert len(test_indexes) == 4 and  len(dev_indexes) == 22
    elif language.startswith("en-rstdt"):
        assert len(dev_indexes) == 35 and len(test_indexes) == 38
    elif language.startswith("en-gum"):
        assert len(test_indexes) == len(dev_indexes) == 24
    elif language.startswith("zh-gcdt"):
        assert len(test_indexes) == len(dev_indexes) == 5
    elif language.startswith("bi-gum"):
        assert len(test_indexes) == len(dev_indexes) == 29
    elif language.startswith("fivebi-gum"):
        assert len(test_indexes) == len(dev_indexes) == 15
    elif language.startswith("fiveen-gum"):
        assert len(test_indexes) == len(dev_indexes) == 10
    elif language.startswith("outbi2zh-gcdt"):
        assert len(test_indexes) == len(dev_indexes) == 5
    elif language.startswith("outfivebi2zh-gcdt"):
        assert len(test_indexes) == len(dev_indexes) == 5
    elif language.startswith("outbi2en-gum"):
        assert len(test_indexes) == len(dev_indexes) == 24
    elif language.startswith("outfivebi2en-gum"):
        assert len(test_indexes) == len(dev_indexes) == 10
    elif language.startswith("outtwelve2fiveen-gum"):
        assert len(test_indexes) == len(dev_indexes) == 10
    elif language.startswith("fivebitransen2en-gum"):
        assert len(test_indexes) == len(dev_indexes) == 10
    elif language.startswith("fivebitransen2zh-gcdt"):
        assert len(test_indexes) == len(dev_indexes) == 5
    elif language.startswith("bitransen2en-gum"):
        assert len(test_indexes) == len(dev_indexes) == 24
    elif language.startswith("bitransen2zh-gcdt"):
        assert len(test_indexes) == len(dev_indexes) == 5
    elif language.startswith("fivebitranszh2en-gum"):
        assert len(test_indexes) == len(dev_indexes) == 10
    elif language.startswith("fivebitranszh2zh-gcdt"):
        assert len(test_indexes) == len(dev_indexes) == 5
    elif language.startswith("bitranszh2en-gum"):
        assert len(test_indexes) == len(dev_indexes) == 24
    elif language.startswith("bitranszh2zh-gcdt"):
        assert len(test_indexes) == len(dev_indexes) == 5
    else:
        assert False

    return dev_indexes, test_indexes, leave_out_indexs

def random_split(folder_path, language):
    FileNames = pickle.load(open(os.path.join(folder_path, "FileName.pickle"), "rb"))
    InputSentences = pickle.load(open(os.path.join(folder_path, "InputSentences.pickle"), "rb"))
    EDUBreaks = pickle.load(open(os.path.join(folder_path, "EDUBreaks.pickle"), "rb"))
    DecoderInput = pickle.load(open(os.path.join(folder_path, "DecoderInputs.pickle"), "rb"))
    RelationLabel = pickle.load(open(os.path.join(folder_path, "RelationLabel.pickle"), "rb"))
    ParsingBreaks = pickle.load(open(os.path.join(folder_path, "ParsingIndex.pickle"), "rb"))
    ParentsIndex = pickle.load(open(os.path.join(folder_path, "ParentsIndex.pickle"), "rb"))
    Sibling = pickle.load(open(os.path.join(folder_path, "Sibling.pickle"), "rb"))
    GoldenMetric = pickle.load(open(os.path.join(folder_path, "GoldenLabelforMetric.pickle"), "rb"))

    sample_number = len(FileNames)
    if "zh-gcdt" in language or "en-gum" in language or "bi-gum" in language or "en-rstdt" in language:
        dev_indexes, test_indexes, leave_out_indexes = find_english_dev_test(FileNames, language)
    else:
        sample_index = [i for i in range(sample_number)]
        random.seed(666)
        dev_test_indexes = sorted(random.sample(sample_index, 0.2*len(FileNames)))
        dev_indexes, test_indexes = dev_test_indexes[:len(dev_test_indexes)//2], dev_test_indexes[len(dev_test_indexes)//2:]
        leave_out_indexes = []
    print(language, "dev: ", dev_indexes, "test: ",  test_indexes, sep="\n")
    
    assert list(set(dev_indexes).intersection(set(test_indexes))) == []

    train_indexs = [i for i in range(sample_number) if i not in dev_indexes + test_indexes + leave_out_indexes]


    Train_FileNames = [item for i, item in enumerate(FileNames) if i in train_indexs]
    Train_InputSentences = [item for i, item in enumerate(InputSentences) if i in train_indexs]
    Train_EDUBreaks = [item for i, item in enumerate(EDUBreaks) if i in train_indexs]
    Train_DecoderInput = [item for i, item in enumerate(DecoderInput) if i in train_indexs]
    Train_RelationLabel = [item for i, item in enumerate(RelationLabel) if i in train_indexs]
    Train_ParsingBreaks = [item for i, item in enumerate(ParsingBreaks) if i in train_indexs]
    Train_ParentsIndex = [item for i, item in enumerate(ParentsIndex) if i in train_indexs]
    Train_Sibling = [item for i, item in enumerate(Sibling) if i in train_indexs]
    Train_GoldenMetric = [item for i, item in enumerate(GoldenMetric) if i in train_indexs]

    Dev_FileNames = [item for i, item in enumerate(FileNames) if i in dev_indexes]
    Dev_InputSentences = [item for i, item in enumerate(InputSentences) if i in dev_indexes]
    Dev_EDUBreaks = [item for i, item in enumerate(EDUBreaks) if i in dev_indexes]
    Dev_DecoderInput = [item for i, item in enumerate(DecoderInput) if i in dev_indexes]
    Dev_RelationLabel = [item for i, item in enumerate(RelationLabel) if i in dev_indexes]
    Dev_ParsingBreaks = [item for i, item in enumerate(ParsingBreaks) if i in dev_indexes]
    Dev_ParentsIndex = [item for i, item in enumerate(ParentsIndex) if i in dev_indexes]
    Dev_Sibling = [item for i, item in enumerate(Sibling) if i in dev_indexes]
    Dev_GoldenMetric = [item for i, item in enumerate(GoldenMetric) if i in dev_indexes]

    Test_FileNames = [item for i, item in enumerate(FileNames) if i in test_indexes]
    Test_InputSentences = [item for i, item in enumerate(InputSentences) if i in test_indexes]
    Test_EDUBreaks = [item for i, item in enumerate(EDUBreaks) if i in test_indexes]
    Test_DecoderInput = [item for i, item in enumerate(DecoderInput) if i in test_indexes]
    Test_RelationLabel = [item for i, item in enumerate(RelationLabel) if i in test_indexes]
    Test_ParsingBreaks = [item for i, item in enumerate(ParsingBreaks) if i in test_indexes]
    Test_ParentsIndex = [item for i, item in enumerate(ParentsIndex) if i in test_indexes]
    Test_Sibling = [item for i, item in enumerate(Sibling) if i in test_indexes]
    Test_GoldenMetric = [item for i, item in enumerate(GoldenMetric) if i in test_indexes]


    save_pickle(Train_FileNames, os.path.join(folder_path, "Training_FileNames.pickle"))
    save_pickle(Train_InputSentences, os.path.join(folder_path, "Training_InputSentences.pickle"))
    save_pickle(Train_EDUBreaks, os.path.join(folder_path, "Training_EDUBreaks.pickle"))
    save_pickle(Train_DecoderInput, os.path.join(folder_path, "Training_DecoderInputs.pickle"))
    save_pickle(Train_RelationLabel, os.path.join(folder_path, "Training_RelationLabel.pickle"))
    save_pickle(Train_ParsingBreaks, os.path.join(folder_path, "Training_ParsingIndex.pickle"))
    save_pickle(Train_ParentsIndex, os.path.join(folder_path, "Training_ParentsIndex.pickle"))
    save_pickle(Train_Sibling, os.path.join(folder_path, "Training_Sibling.pickle"))
    save_pickle(Train_GoldenMetric, os.path.join(folder_path, "Training_GoldenLabelforMetric.pickle"))

    save_pickle(Dev_FileNames, os.path.join(folder_path, "Deving_FileNames.pickle"))
    save_pickle(Dev_InputSentences, os.path.join(folder_path, "Deving_InputSentences.pickle"))
    save_pickle(Dev_EDUBreaks, os.path.join(folder_path, "Deving_EDUBreaks.pickle"))
    save_pickle(Dev_DecoderInput, os.path.join(folder_path, "Deving_DecoderInputs.pickle"))
    save_pickle(Dev_RelationLabel, os.path.join(folder_path, "Deving_RelationLabel.pickle"))
    save_pickle(Dev_ParsingBreaks, os.path.join(folder_path, "Deving_ParsingIndex.pickle"))
    save_pickle(Dev_ParentsIndex, os.path.join(folder_path, "Deving_ParentsIndex.pickle"))
    save_pickle(Dev_Sibling, os.path.join(folder_path, "Deving_Sibling.pickle"))
    save_pickle(Dev_GoldenMetric, os.path.join(folder_path, "Deving_GoldenLabelforMetric.pickle"))

    save_pickle(Test_FileNames, os.path.join(folder_path, "Testing_FileNames.pickle"))
    save_pickle(Test_InputSentences, os.path.join(folder_path, "Testing_InputSentences.pickle"))
    save_pickle(Test_EDUBreaks, os.path.join(folder_path, "Testing_EDUBreaks.pickle"))
    save_pickle(Test_DecoderInput, os.path.join(folder_path, "Testing_DecoderInputs.pickle"))
    save_pickle(Test_RelationLabel, os.path.join(folder_path, "Testing_RelationLabel.pickle"))
    save_pickle(Test_ParsingBreaks, os.path.join(folder_path, "Testing_ParsingIndex.pickle"))
    save_pickle(Test_ParentsIndex, os.path.join(folder_path, "Testing_ParentsIndex.pickle"))
    save_pickle(Test_Sibling, os.path.join(folder_path, "Testing_Sibling.pickle"))
    save_pickle(Test_GoldenMetric, os.path.join(folder_path, "Testing_GoldenLabelforMetric.pickle"))



def split_train_test(base_path):
    translated_folders = sorted(glob(base_path + 'to*'))
    for translated_folder in translated_folders:
        language_folders = sorted(glob(translated_folder+'/*'))
        for language_folder in language_folders:
            language = language_folder.split('/')[-1]
            # if not language.startswith("en-rstdt-"):  # Logan: specific dataset
            if "4news" not in language:  # Logan: specific dataset
                continue
            random_split(language_folder, language)


if __name__ == "__main__":
    base_path = '../../data/pickle-data/depth/'
    split_train_test(base_path)







