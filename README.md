# GCDT: Georgetown Chinese Discourse Treebank

GCDT is the largest (as of October 2022) hierarchical discourse treebank for Mandarin Chinese in the framework of Rhetorical Structure Theory (RST). 

GCDT covers over 60K tokens across five genres of freely available text, using the same relation inventory as the contemporary RST treebank for English -- GUM (https://github.com/amir-zeldes/gum). 


## Corpus statistics 
The corpus contains 50 documents, 10 from each of the five genres below.

In total, there are 62,905 tokens and  9,717 EDUs. 

We provide an 8:1:1 train:dev:test split for each genre as shown in the table below.

| Genre   |  #docs |  Avg #tokens/doc | Avg #EDUs/doc | Source | Dev doc | Test doc |
|:----------:|:-------------:|------:|------:|:-------------:|:-------------:|:-------------:|
| academic | 10 | 1416.8 | 203.3 | https://www.hanspub.org/ | gcdt_academic_peoples | gcdt_academic_dingzhen |
| bio | 10 | 1348.5 | 202.1 | https://zh.wikipedia.org/ | gcdt_bio_byron | gcdt_bio_dvorak |
| interview | 10 | 1146.4 | 181.2 | https://zh.wikinews.org/ | gcdt_interview_ward | gcdt_interview_wimax |
| news | 10 | 1124.9 | 165.2 | https://zh.wikinews.org/ | gcdt_news_famine | gcdt_news_simplified |
| wikihow | 10 | 1253.9 | 219.9 | https://zh.wikihow.com/ | gcdt_whow_hiking | gcdt_whow_thanksgiving |



## Citing

To cite this corpus, please refer to the following article:

TODO

```
TODO
```

To cite the Chinese RST annotation guideline, please cite the following:

TODO

```
TODO
```

## Data and utils
Please see the two sub-directories for annotated data and relevant scripts.

- data/  https://github.com/logan-siyao-peng/GCDT/tree/main/data
- utils/ https://github.com/logan-siyao-peng/GCDT/tree/main/utils


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

