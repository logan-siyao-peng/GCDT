# GCDT Data Directory

We provide the following data formats for our GCDT corpus.
In each format, files are separated into train/, dev/ and test/ (and double/ if applicable).

* raw/ - raw texts from source websites with metadata, XML and paragraph segmentation annotations
* tokenized/ - manually tokenized texts with one sentence per line according to "The Segmentation Guidelines for the Penn Chinese Treebank (3.0)"
* conllu/ - automatically stanza parsed Universal Dependencies from gold tokenized/ using https://github.com/logan-siyao-peng/GCDT/blob/main/utils/parse_dependencies.py
* rs3/ - constituency styled RST annotations for 50 documents and 5 additional double annotations on the test set
* rs3_extracted_edus/ - text files with one EDU per line automatically extracted from annotated rs3 files using https://github.com/logan-siyao-peng/GCDT/blob/main/utils/extract_edu_from_rs3.py 
* autotrans_rs3/ - constituency styled rs3 files with automatic EDU-level Chinese --> English Google translations appended to the end of each EDU using https://github.com/logan-siyao-peng/GCDT/blob/main/utils/autotrans_rs3.py
* autotrans_extracted_edus	- one EDU per line automatically extracted from translation-appended rs3 files using https://github.com/logan-siyao-peng/GCDT/blob/main/utils/extract_edu_from_rs3.py
* rsd - automatically coverted from rs3 files using https://github.com/amir-zeldes/gum/blob/master/_build/utils/rst2dep.py
* dis - automatically coverted from rs3 files using https://github.com/amir-zeldes/gum/blob/master/_build/utils/rst2dis.py
* others - EDU segmentations from the second annotator before adjudication
			
