This code is adapted from https://github.com/seq-to-mind/DMRST_Parser/.

Please cite their papers if you find the code useful. 


## Major changes from the original DMRST code in terms of data preprocessing:
- Different relation classes for GUM/GCDT when compared to RST-DT (1_uniform_treebanks/code/src/relationSet.py, 1_uniform_treebanks/code/src/data.py, 2_convert_to_our_format/1_MUL_generate_input_pkl_XLMR.py)
- Experimented with multiple language models on each dataset (2_convert_to_our_format/1_MUL_generate_input_pkl_XLMR.py)
- Specifies train/dev/test splits for different datasets as well as different combinations (2_convert_to_our_format/2_split_train_test_pickle.py)


## Preprocessing steps

### Step 1: Convert rs3 or dis data into rst-sample-dmrg (separate files for EDUs and relations)

Here are the imput arguments:

- treebank: the directory of your individual rs3 or dis files
- outpath: the directory of the saved dmrg files
- format: whether the original annotations are rs3, dis
- mapping: use gum_classes for GUM and GCDT; rstdt_classes for RST-DT

```
cd 1_uniform_treebanks/code/src/

python dt_reader.py --treebank ../../../../data/rs3/zh-gcdt-20220625/ --outpath ../../../../data/rst-sample-dmrg/zh-gcdt/ --format rs3 --mapping gum_classes
```


### Step 2: Encode documents with language models

This step encodes all corpora in the rst-sample-dmrg/ directory with language models specified in 1_MUL_generate_input_pkl_XLMR.py and saves them as pickled data. 


```
cd ../../../2_convert_to_our_format/

python 1_MUL_generate_input_pkl_XLMR.py
```

### Step 3: Splits pickled data based on train/dev/test document lists

This step splits the pickled data based on specified train/dev/test splits. 

```
python 2_split_train_test_pickle.py
```
