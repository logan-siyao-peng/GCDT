# GCDT: Georgetown Chinese Discourse Treebank

GCDT is the largest (as of October 2022) hierarchical discourse treebank for Mandarin Chinese in the framework of Rhetorical Structure Theory (RST). 

GCDT covers over 60K tokens across five genres of freely available text, using the same relation inventory as the contemporary RST treebank for English -- GUM (https://github.com/amir-zeldes/gum). 


## Corpus statistics 
The corpus contains 50 documents, 10 from each of the five genres below.

In total, there are 62,905 tokens and  9,717 EDUs. 

We provide an 8:1:1 train:dev:test split for each genre as shown in the table below.

The five test documents are double annotated and the second annotation version appears in the double/ partition.

| Genre   |  #docs |  Avg #tokens/doc | Avg #EDUs/doc | Source | Dev doc | Test doc |
|:----------:|:-------------:|------:|------:|:-------------:|:-------------:|:-------------:|
| academic | 10 | 1416.8 | 203.3 | https://www.hanspub.org/ | gcdt_academic_peoples | gcdt_academic_dingzhen |
| bio | 10 | 1348.5 | 202.1 | https://zh.wikipedia.org/ | gcdt_bio_byron | gcdt_bio_dvorak |
| interview | 10 | 1146.4 | 181.2 | https://zh.wikinews.org/ | gcdt_interview_ward | gcdt_interview_wimax |
| news | 10 | 1124.9 | 165.2 | https://zh.wikinews.org/ | gcdt_news_famine | gcdt_news_simplified |
| wikihow | 10 | 1253.9 | 219.9 | https://zh.wikihow.com/ | gcdt_whow_hiking | gcdt_whow_thanksgiving |



## Citing

Please cite the following for the source paper -- [GCDT: A Chinese RST Treebank for Multigenre and Multilingual Discourse Parsing](https://aclanthology.org/2022.aacl-short.47/):

```
@inproceedings{peng_gcdt_2022,
	address = {Online only},
	title = {{GCDT}: {A} {Chinese} {RST} {Treebank} for {Multigenre} and {Multilingual} {Discourse} {Parsing}},
	shorttitle = {{GCDT}},
	url = {https://aclanthology.org/2022.aacl-short.47},
	abstract = {A lack of large-scale human-annotated data has hampered the hierarchical discourse parsing of Chinese. In this paper, we present GCDT, the largest hierarchical discourse treebank for Mandarin Chinese in the framework of Rhetorical Structure Theory (RST). GCDT covers over 60K tokens across five genres of freely available text, using the same relation inventory as contemporary RST treebanks for English. We also report on this dataset's parsing experiments, including state-of-the-art (SOTA) scores for Chinese RST parsing and RST parsing on the English GUM dataset, using cross-lingual training in Chinese and English with multilingual embeddings.},
	urldate = {2022-11-22},
	booktitle = {Proceedings of the 2nd {Conference} of the {Asia}-{Pacific} {Chapter} of the {Association} for {Computational} {Linguistics} and the 12th {International} {Joint} {Conference} on {Natural} {Language} {Processing}},
	publisher = {Association for Computational Linguistics},
	author = {Peng, Siyao and Liu, Yang Janet and Zeldes, Amir},
	month = nov,
	year = {2022},
	pages = {382--391},
	file = {Full Text PDF:/Users/loganpeng/Zotero/storage/IEPKWVJH/Peng et al. - 2022 - GCDT A Chinese RST Treebank for Multigenre and Mu.pdf:application/pdf},
}
```

Please cite the following for the [Chinese Discourse Annotation Reference Manual](https://hal.archives-ouvertes.fr/hal-03821884):


```

@techreport{peng_chinese_2022,
	type = {Research {Report}},
	title = {Chinese {Discourse} {Annotation} {Reference} {Manual}},
	url = {https://hal.archives-ouvertes.fr/hal-03821884},
	abstract = {This document provides extensive guidelines and examples for Rhetorical Structure Theory (RST) annotation in Mandarin Chinese. The guideline is divided into three sections. We first introduce preprocessing steps to prepare data for RST annotation. Secondly, we discuss syntactic criteria to segment texts into Elementary Discourse Units (EDUs). Lastly, we provide examples to define and distinguish discourse relations in different genres. We hope that this reference manual can facilitate RST annotations in Chinese and accelerate the development of the RST framework across languages.},
	urldate = {2022-11-30},
	institution = {Georgetown University (Washington, D.C.)},
	author = {Peng, Siyao and Liu, Yang Janet and Zeldes, Amir},
	month = oct,
	year = {2022},
	keywords = {Chinese, Discourse Analysis Representation, Rhetorical Structure Theory RST},
	file = {HAL PDF Full Text:/Users/loganpeng/Zotero/storage/Q7ZFHQ3Q/Peng et al. - 2022 - Chinese Discourse Annotation Reference Manual.pdf:application/pdf},
}
```

## Data, utils and adapted DMRST parser
Please see the sub-directories for annotated data, scripts, and adapted DMRST parser.

- data/  https://github.com/logan-siyao-peng/GCDT/tree/main/data
- utils/ https://github.com/logan-siyao-peng/GCDT/tree/main/utils
- adapted_DMRST/ https://github.com/logan-siyao-peng/GCDT/tree/main/adapted_DMRST

## Data and metadata annotations
- XML and metadata annotations -- gold
- Paragraph and sentence splits -- gold
- Tokenization -- gold
- Dependency parses -- predicted by stanza
- rs3 to rsd, dis conversions -- automatic
- EDU-wise Chinese --> English translation -- automatic


## Relation distributions in GCDT
We present GCDT's distribution of the 32 discourse relations in the table below.

| relation name | total counts | percentage |
|:----------:|------:|------:|
| joint-list | 2808 | 22.16 |
| same-unit | 2375 | 18.74 |
| elaboration-attribute | 992 | 7.83 |
| joint-sequence | 647 | 5.10 |
| joint-other | 595 | 4.69 |
| elaboration-additional | 551 | 4.35 |
| attribution-positive | 551 | 4.35 |
| explanation-evidence | 522 | 4.12 |
| adversative-contrast | 412 | 3.25 |
| context-circumstance | 350 | 2.76 |
| context-background | 350 | 2.76 |
| organization-preparation | 288 | 2.27 |
| causal-cause | 238 | 1.88 |
| contingency-condition | 228 | 1.80 |
| organization-heading | 225 | 1.78 |
| adversative-concession | 208 | 1.64 |
| purpose-goal | 193 | 1.52 |
| restatement-partial | 164 | 1.29 |
| evaluation-comment | 136 | 1.07 |
| mode-means | 132 | 1.04 |
| causal-result | 123 | 0.97 |
| explanation-justify | 120 | 0.95 |
| joint-disjunction | 92 | 0.73 |
| adversative-antithesis | 78 | 0.62 |
| mode-manner | 75 | 0.59 |
| topic-question | 57 | 0.45 |
| restatement-repetition | 40 | 0.32 |
| explanation-motivation | 35 | 0.28 |
| organization-phatic | 33 | 0.26 |
| attribution-negative | 29 | 0.23 |
| purpose-attribute | 26 | 0.21 |
| topic-solutionhood | 1 | 0.01 |

