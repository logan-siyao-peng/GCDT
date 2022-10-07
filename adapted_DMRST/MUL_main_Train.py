import re
import os
import pickle
import torch
import numpy as np
import random
import argparse
from Training import Train, EvalOnly
import time
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, MBart50Tokenizer
import config
from model_depth import ParsingNet

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.global_gpu_id)
base_path = config.tree_infer_mode + "_mode/"

def parse_args():
    parser = argparse.ArgumentParser(description='RSTParser')
    parser.add_argument('--GPUforModel', type=int, default=config.global_gpu_id, help='Which GPU to run')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size') # default=3
    parser.add_argument('--eval_size', type=int, default=1,
                        help='Evaluation size')
    parser.add_argument('--hidden_size', type=int, default=config.hidden_size, help='Hidden size of RNN')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--dropout_e', type=float, default=0.5, help='Dropout rate for encoder')
    parser.add_argument('--dropout_d', type=float, default=0.5, help='Dropout rate for decoder')
    parser.add_argument('--dropout_c', type=float, default=0.5, help='Dropout rate for classifier')
    parser.add_argument('--input_is_word', type=str, default='True', help='Whether the encoder input is word or EDU')

    parser.add_argument('--atten_model', choices=['Dotproduct', 'Biaffine'], default='Dotproduct', help='Attention mode')
    parser.add_argument('--classifier_input_size', type=int, default=config.hidden_size, help='Input size of relation classifier')
    parser.add_argument('--classifier_hidden_size', type=int, default=int(config.hidden_size / 1), help='Hidden size of relation classifier')
    parser.add_argument('--classifier_bias', type=str, default='True', help='Whether classifier has bias')
    parser.add_argument('--seed', type=int, default=111, help='Seed number')
    
    parser.add_argument('--epoch', type=int, default=50, help='Epoch number')

    parser.add_argument('--lr', type=float, default=0.00002, help='Initial lr')
    parser.add_argument('--lr_decay_epoch', type=int, default=1, help='Lr decay epoch')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay rate')

    parser.add_argument('--datapath', type=str, default="./data/pickle-data/depth/to_pt/zh-gcdt-hfl-chinese-roberta-wwm-ext/", help='Data path')
    parser.add_argument('--savepath', type=str, default=None, help='Model save path')
    parser.add_argument('--finetuning', type=str, default="False", choices=["True", "False"],
                        help='Whether to finetune on a saved model or not')
    parser.add_argument('--use_org_Parseval', choices=['True', 'False'], default='True')
    parser.add_argument('--eval_only', choices=['True', 'False'], default='False')
    args = parser.parse_args()
    print('args.use_org_Parseval:', str(args.use_org_Parseval))

    return args


if __name__ == '__main__':

    args = parse_args()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.GPUforModel) if USE_CUDA else "cpu")

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    rnn_layers = args.rnn_layers
    dropout_e = args.dropout_e
    dropout_d = args.dropout_d
    dropout_c = args.dropout_c
    input_is_word = args.input_is_word
    atten_model = args.atten_model
    classifier_input_size = args.classifier_input_size
    classifier_hidden_size = args.classifier_hidden_size
    classifier_bias = args.classifier_bias

    seednumber = args.seed
    data_path = args.datapath
    data_path_split = [x for x in data_path.split(os.sep) if x != ""]
    data_base = data_path_split[-1]
    if args.savepath == None and args.finetuning == "True":
        assert False
    elif args.savepath == None and args.finetuning == "False":
        save_path = base_path + "Savings/%s_bs%d_seed%d/" % (data_base, batch_size, seednumber)
        finetune_frompath = None
    elif args.savepath != None and args.finetuning == "True":
        finetune_frompath = base_path + args.savepath
        save_path = finetune_frompath.replace("Savings/", "Savings/finetuned_")
    elif args.savepath != None and args.finetuning == "False":
        finetune_frompath = None
        save_path = base_path + args.savepath
        
    eval_size = args.eval_size
    epoch = args.epoch
    lr = args.lr
    lr_decay_epoch = args.lr_decay_epoch
    weight_decay = args.weight_decay
    
    # get language embedding
    language_embedding_name = re.sub(r".*gcdt-", "", (re.sub(r".*rstdt-", "", re.sub(r".*gum-", "", data_base))))
    language_embedding_name = language_embedding_name.replace("hfl-", "hfl/").replace("SpanBERT-", "SpanBERT/")

    """ BERT tokenizer and model """
    print("language embedding name: ", language_embedding_name)
    bert_tokenizer = AutoTokenizer.from_pretrained(language_embedding_name, use_fast=True)
    bert_model = AutoModel.from_pretrained(language_embedding_name)
    print("Language embedding loaded from pretrained: ", language_embedding_name)

    """ freeze some layers """
    for name, param in bert_model.named_parameters():
        layer_num = re.findall("layer\.(\d+)\.", name)
        if len(layer_num) > 0 and int(layer_num[0]) > 2:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if USE_CUDA:
        language_model = bert_model.cuda()
    else:
        language_model = bert_model

    # Setting random seeds 
    torch.manual_seed(seednumber)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seednumber)
    np.random.seed(seednumber)
    random.seed(seednumber)

    # Process bool args       
    if args.classifier_bias == 'True':
        classifier_bias = True

    elif args.classifier_bias == 'False':
        classifier_bias = False

    Tr_InputSentences = []
    Tr_EDUBreaks = []
    Tr_DecoderInput = []
    Tr_RelationLabel = []
    Tr_ParsingBreaks = []
    Tr_GoldenMetric = []
    Tr_ParentsIndex = []
    Tr_SiblingIndex = []

    Dev_InputSentences = []
    Dev_EDUBreaks = []
    Dev_DecoderInput = []
    Dev_RelationLabel = []
    Dev_ParsingBreaks = []
    Dev_GoldenMetric = []
    Dev_ParentsIndex = []
    Dev_SiblingIndex = []

    # Load Testing data
    Test_InputSentences = []
    Test_EDUBreaks = []
    Test_DecoderInput = []
    Test_RelationLabel = []
    Test_ParsingBreaks = []
    Test_GoldenMetric = []

    # Load Training data
    Tr_InputSentences.extend(pickle.load(open(os.path.join(data_path, "Training_InputSentences.pickle"), "rb")))
    Tr_EDUBreaks.extend(pickle.load(open(os.path.join(data_path, "Training_EDUBreaks.pickle"), "rb")))
    Tr_DecoderInput.extend(pickle.load(open(os.path.join(data_path, "Training_DecoderInputs.pickle"), "rb")))
    Tr_RelationLabel.extend(pickle.load(open(os.path.join(data_path, "Training_RelationLabel.pickle"), "rb")))
    Tr_ParsingBreaks.extend(pickle.load(open(os.path.join(data_path, "Training_ParsingIndex.pickle"), "rb")))
    Tr_GoldenMetric.extend(pickle.load(open(os.path.join(data_path, "Training_GoldenLabelforMetric.pickle"), "rb")))
    Tr_ParentsIndex.extend(pickle.load(open(os.path.join(data_path, "Training_ParentsIndex.pickle"), "rb")))
    Tr_SiblingIndex.extend(pickle.load(open(os.path.join(data_path, "Training_Sibling.pickle"), "rb")))

    # Load Deving data
    Dev_InputSentences.extend(pickle.load(open(os.path.join(data_path, "Deving_InputSentences.pickle"), "rb")))
    Dev_EDUBreaks.extend(pickle.load(open(os.path.join(data_path, "Deving_EDUBreaks.pickle"), "rb")))
    Dev_DecoderInput.extend(pickle.load(open(os.path.join(data_path, "Deving_DecoderInputs.pickle"), "rb")))
    Dev_RelationLabel.extend(pickle.load(open(os.path.join(data_path, "Deving_RelationLabel.pickle"), "rb")))
    Dev_ParsingBreaks.extend(pickle.load(open(os.path.join(data_path, "Deving_ParsingIndex.pickle"), "rb")))
    Dev_GoldenMetric.extend(pickle.load(open(os.path.join(data_path, "Deving_GoldenLabelforMetric.pickle"), "rb")))
    Dev_ParentsIndex.extend(pickle.load(open(os.path.join(data_path, "Deving_ParentsIndex.pickle"), "rb")))
    Dev_SiblingIndex.extend(pickle.load(open(os.path.join(data_path, "Deving_Sibling.pickle"), "rb")))

    # Load Testing data
    Test_InputSentences.extend(pickle.load(open(os.path.join(data_path, "Testing_InputSentences.pickle"), "rb")))
    Test_EDUBreaks.extend(pickle.load(open(os.path.join(data_path, "Testing_EDUBreaks.pickle"), "rb")))
    Test_DecoderInput.extend(pickle.load(open(os.path.join(data_path, "Testing_DecoderInputs.pickle"), "rb")))
    Test_RelationLabel.extend(pickle.load(open(os.path.join(data_path, "Testing_RelationLabel.pickle"), "rb")))
    Test_ParsingBreaks.extend(pickle.load(open(os.path.join(data_path, "Testing_ParsingIndex.pickle"), "rb")))
    Test_GoldenMetric.extend(pickle.load(open(os.path.join(data_path, "Testing_GoldenLabelforMetric.pickle"), "rb")))

    # To check data
    sent_temp = ''
    print("Checking Data...")
    for word_temp in Tr_InputSentences[2]:
        sent_temp = sent_temp + ' ' + word_temp
    print(sent_temp)
    print('... ...')
    print("That's great! No error found!")
    print("All train sample number:", len(Tr_InputSentences))

    # To save model and data
    FileName = str(seednumber) + "_" + config.tree_infer_mode + '_Batch_' + str(batch_size) + 'Hidden_' + str(hidden_size) + \
               'LR' + str(lr) + "_" + str(time.time())

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    print("Model save path", save_path)
    print("Finetune from path", finetune_frompath)
        

    """ relation number is set at 42 """
    if "en-rstdt" in data_path:
        number_of_relations = 42
    elif "-gum" in data_path or "-gcdt" in data_path:
        number_of_relations = 30
    model = ParsingNet(language_model, hidden_size, hidden_size,
                       hidden_size, atten_model, classifier_input_size, classifier_hidden_size, number_of_relations,
                       classifier_bias, rnn_layers, dropout_e, dropout_d, dropout_c, bert_tokenizer=bert_tokenizer)

    if USE_CUDA:
        model = model.cuda()


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
    # Train
    if args.eval_only == 'False':
        print("Total trainable parameter number is: ", count_parameters(model))
    
        TrainingProcess = Train(model, 
                                Tr_InputSentences, Tr_EDUBreaks, Tr_DecoderInput,
                                Tr_RelationLabel, Tr_ParsingBreaks, Tr_GoldenMetric,
                                Tr_ParentsIndex, Tr_SiblingIndex,
                                Dev_InputSentences, Dev_EDUBreaks, Dev_DecoderInput,
                                Dev_RelationLabel, Dev_ParsingBreaks, Dev_GoldenMetric,
                                Dev_ParentsIndex, Dev_SiblingIndex,
                                Test_InputSentences, Test_EDUBreaks, Test_DecoderInput,
                                Test_RelationLabel, Test_ParsingBreaks, Test_GoldenMetric,
                                batch_size, eval_size, epoch, lr, lr_decay_epoch,
                                weight_decay,
                                save_path, finetune_frompath, args.use_org_Parseval=='True')
    
        best_epoch_Dev, best_F_relation_Dev, best_P_relation_Dev, best_R_relation_Dev, best_F_span_Dev, \
        best_P_span_Dev, best_R_span_Dev, best_F_nuclearity_Dev, best_P_nuclearity_Dev, \
        best_R_nuclearity_Dev = TrainingProcess.train()
    
        print('--------------------------------------------------------------------')
        print('Training Completed!')
        print('Processing...')
        print('The best Dev epoch is: ', best_epoch_Dev)
        print('The best Dev F1 points for Span Nuclearity Relation are:\n %.4f\t%.4f\t%.4f'
              % (best_F_span_Dev, best_F_nuclearity_Dev, best_F_relation_Dev))
        # Save result
        with open(os.path.join(save_path, 'Results.csv'), 'a') as f:
            f.write(FileName + ',' + ','.join(map(str, [best_epoch_Dev, best_F_relation_Dev,
                                                        best_P_relation_Dev, best_R_relation_Dev, best_F_span_Dev,
                                                        best_P_span_Dev, best_R_span_Dev, best_F_nuclearity_Dev,
                                                        best_P_nuclearity_Dev, best_R_nuclearity_Dev])) + '\n')
        
    
    print("Data path: %s\nModel path: %s" % (data_path, save_path))
    EvalProcess = EvalOnly(model,
                 Test_InputSentences, Test_EDUBreaks, Test_DecoderInput,
                 Test_RelationLabel, Test_ParsingBreaks, Test_GoldenMetric,
                 eval_size,
                 save_path, args.use_org_Parseval=='True')
    
    EvalProcess.evaluate()

