# GUM_Chinese

A GUM-like RST dataset for Chinese

50 documents are collected with 10 from each of the five genres.

63,000 tokens (updated on Jan 12 after gold tokenization)


## Five Genres
- academic (source: https://www.hanspub.org/)
- bio (source: https://zh.wikipedia.org/)
- interview (source: https://zh.wikinews.org/)
- news (source: https://zh.wikinews.org/)
- wikihow (source: https://zh.wikihow.com/)

## Corpus Statistics

### Total
- 62908 tokens
- 9842 EDUs


### By Genre
| Genre   |  Number of documents |  Number of tokens per document | Number of EDUs per document |
|:----------:|:-------------:|------:|------:|
| academic | 1416.9 | 204.9 |
| bio |    1348.6 |	206 |
| interview | 1146.5 | 183.2 |
| news | 1124.9 | 166 |
| wikihow | 1253.9 | 224.1 |


### relation distribution
TODO: check which ones to include 

| relation name | total counts | percentage |
| joint-list | 28 d65 | 22.32 |
| same-unit | 2402 | 18.71 |
| elaboration-attribute | 970 | 7.56 |
| joint-sequence | 628 | 4.89 |
| joint-other | 606 | 4.72 |
| attribution-positive | 575 | 4.48 |
| elaboration-additional | 538 | 4.19 |
| explanation-evidence | 528 | 4.11 |
| adversative-contrast | 428 | 3.33 |
| context-circumstance | 393 | 3.06 |
| context-background | 367 | 2.86 |
| organization-preparation | 299 | 2.33 |
| causal-cause | 252 | 1.96 |
| organization-heading | 223 | 1.74 |
| contingency-condition | 219 | 1.71 |
| adversative-concession | 214 | 1.67 |
| purpose-goal | 196 | 1.53 |
| restatement-partial | 180 | 1.40 |
| evaluation-comment | 151 | 1.18 |
| mode-means | 139 | 1.08 |
| causal-result | 115 | 0.90 |
| explanation-justify | 108 | 0.84 |
| joint-disjunction | 84 | 0.65 |
| adversative-antithesis | 73 | 0.57 |
| mode-manner | 67 | 0.52 |
| topic-question | 57 | 0.44 |
| restatement-repetition | 38 | 0.30 |
| organization-phatic | 34 | 0.26 |
| attribution-negative | 32 | 0.25 |
| purpose-attribute | 28 | 0.22 |
| explanation-motivation | 24 | 0.19 |
| topic-solutionhood | 4 | 0.03 |
| cause-result | 1 | 0.01 |

## Preprocessing Steps
- XML and metadata annotations (gold)
- Paragraph and sentence splits (gold)
- Tokenization (gold) 
- Dependency parses (predicted by stanza)

## RST annotations
Guideline: see https://docs.google.com/document/d/1OmeqkDIYg5IM_pmULMzDJi__FAe7kzJmMdLtZxxd1LE/edit?usp=sharing

