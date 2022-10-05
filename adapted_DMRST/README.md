# Adapted DMRST Parser

## Introduction

This directory is forked and adapted from the original DMRST parser: https://github.com/seq-to-mind/DMRST_Parser.

Please cite their paper:

`DMRST: A Joint Framework for Document-Level Multilingual RST Discourse Segmentation and Parsing` and `Multilingual Neural RST Discourse Parsing.`


## Sample commands

### To train the model

Please specify your --datapath, and your --savepath would be Savings/{corpus\_embedding\_name}\_bs{batch\_size}\_seed\_{seed_number} by default.

`python MUL_main_Train.py --datapath=./data/pickle-data/depth/to_pt/zh-gcdt-hfl-chinese-roberta-wwm-ext/  --use_org_Parseval True --epoch 15 --batch_size 1 --seed 111`

### To evaluate with saved torch dumps

Please specify your --datapath and --savepath:

`python MUL_main_Train.py --use_org_Parseval True --eval_only True --datapath=./data/pickle-data/depth/to_pt/zh-gcdt-hfl-chinese-roberta-wwm-ext/ --savepath=Savings/zh-gcdt-roberta-basehfl-chinese-roberta-wwm-ext_bs1_seed111/`


## Major changes from orginal DMRST code:

- Different relation classes used for GUM/GCDT versus RST-DT (MUL_main_train.py, DataHandler.py, data/rstdt-relations.txt, gum-relations.txt)
- Allow different language models (MUL_main_train.py, Preprocess_RST_Data/)
- Allow finetuning on existing torch dumps (MUL_main_train.py)
- Specify Train/Dev/Test three-way splits (MUL_main_train.py, Training.py)
- Choose the best epoch based on Dev Relation Score and save the current best torch dump (Training.py)
- Allow running (especially) testing on CPUs (most python files)
 

<hr>
<br>
<br>
<br>

# Original DMRST Parser Readme:

## Introduction
* One implementation of the paper `DMRST: A Joint Framework for Document-Level Multilingual RST Discourse Segmentation and Parsing` and `Multilingual Neural RST Discourse Parsing.`
 <br>
* Users can apply it to parse the input text from scratch, and get the EDU segmentations and the parsed tree structure. <br>
* The model supports both sentence-level and document-level RST discourse parsing. <br>
* This repo and the pre-trained model are only for research use. Please cite the papers if they are helpful. <br>

## Package Requirements
The model training and inference scripts were tested on following libraries and versions:
1. pytorch==1.7.1
2. transformers==4.8.2

## Training: How to convert treebanks to our format for this framework
* Following steps in the two sub-folders under `Preprocess_RST_Data`.
* Note that the `XLM-Roberta-base tokenizer` is used in both treebank pre-processing and model training scripts. For other tokenizers, you should change them accordingly.
* After all treebank pre-processing steps, all samples will be stored in pickle files (the output path is set by user).
* Since some treebanks need LDC license, here we only provide one public dataset as example.
* Tne example pre-processed treebank GUM (Zeldes, A., 2017) (English-only) is located at the folder `./depth_mode/pkl_data_for_train/en-gum/`.

## Training: How to train a model with a pre-processed treebank
* Run the script `MUL_main_Train.py` to train a model.  
* Before you start to train, we recommend that you read the parameter settings. 
* The pre-processed data in folder `./depth_mode/pkl_data_for_train/en-gum/` (English-only) will be used for training by default.
* Note that the `XLM-Roberta-base tokenizer` is used in both treebank pre-processing and model training scripts. For other tokenizers, you should change them accordingly.

## Inference: Supported Languages
Instead of re-training the model, you can use the well-trained parser for inference (model checkpoint is located at `./depth_mode/Savings/`). <br>
We trained and evaluated the model with the multilingual collection of RST discourse treebanks, and it natively supports 6 languages: English, Portuguese, Spanish, German, Dutch, Basque. Interested users can also try other languages.

## Inference: Data Format
* [Input] `InputSentence`: The input document/sentence, and the raw text will be tokenizaed and encoded by the `xlm-roberta-base` language backbone. <br>
    * Raw Sequence Example: <br>
    *Although the report, which has released before the stock market opened, didn't trigger the 190.58 point drop in the Dow Jones Industrial Average, analysts said it did play a role in the market's decline.* <br>

* [Output] `EDU_Breaks`: The indices of the EDU boundary tokens, including the last word of the sentence. <br>
    * Output Example: [5, 10, 17, 33, 37, 49] <br>
    * Segmented Sequence Example ('||' denotes the EDU boundary positions for better readability):  <br>
    *Although the report, || which has released || before the stock market opened, || didn't trigger the 190.58 point drop in the Dow Jones Industrial Average, || analysts said || it did play a role in the market's decline. ||* <br>

* [Output] `tree_parsing_output`: The model outputs of the discourse parsing tree follow this top-down constituency parsing format. <br>
   * (1:Satellite=Contrast:4,5:Nucleus=span:6) (1:Nucleus=Same-Unit:3,4:Nucleus=Same-Unite:4) (5:Satellite=Attribution:5,6:Nucleus=span:6) (1:Satellite=span:1,2:Nucleus=Elaboration:3) (2:Nucleus=span:2,3:Satellite=Temporal:3) <br>

## Inference: How to use it for parsing
* Put the text paragraph to the file `./data/text_for_inference.txt`. <br>
* Pre-trained model checkpoint is located at `./depth_mode/Savings/`. <br>
* Run the script `MUL_main_Infer.py` to obtain the RST parsing result. See the script for detailed model output. <br>
* We recommend users to run the parser on a GPU-equipped environment. <br>

## Citation
```
@inproceedings{liu-etal-2021-dmrst,
    title = "{DMRST}: A Joint Framework for Document-Level Multilingual {RST} Discourse Segmentation and Parsing",
    author = "Liu, Zhengyuan and Shi, Ke and Chen, Nancy",
    booktitle = "Proceedings of the 2nd Workshop on Computational Approaches to Discourse",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic and Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.codi-main.15",
    pages = "154--164",
}
```
```
@inproceedings{liu2020multilingual,
  title={Multilingual Neural RST Discourse Parsing},
  author={Liu, Zhengyuan and Shi, Ke and Chen, Nancy},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={6730--6738},
  year={2020}
}
```

