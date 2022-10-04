# GCDT Utils Directory

The following scripts are used to maintain our GCDT corpus:

* requirements.txt - essential pip packages

* extract_edu_from_rs3.py - extracts EDU from rs3 files. By default the language setting (-l) is Chinese (zh); if you would like to parse the auto-translated rs3 files, pass the option "-l bi" to the python command.

* autotrans_rs3.py - translates Chinese EDUs to English using Google Translate and appends the translations to the end of each EDU

* parse_dependencies.py - parses Universal Dependencies from gold tokenized files using stanza. You would need to download the "zh-hans" package from stanza as well as the OntoNotes-trained pickle files for better postagging and parsing performances https://drive.google.com/drive/folders/1HfjTg0CfMCBDflEYK5Ddo1Jhbkc3Szv_?usp=sharing


* compute_IAA.py - computes IAA for EDU segmentation		

* validate_corpus.py - validates the corpus to ensure consistent tokenization, valid xml tags, etc.

* utils.py - some basic data reading and writting scripts
		


