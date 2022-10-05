This code is adapted from https://github.com/seq-to-mind/DMRST_Parser/.

Please cite their papers if you find the code useful. 


## Major changes from the original DMRST code in terms of data preprocessing:
- Different relation classes for GUM/GCDT when compared to RST-DT (1_uniform_treebanks/code/src/relationSet.py, 1_uniform_treebanks/code/src/data.py, 2_convert_to_our_format/1_MUL_generate_input_pkl_XLMR.py)
- Experimented with multiple language models on each dataset (2_convert_to_our_format/1_MUL_generate_input_pkl_XLMR.py)
- Specifies train/dev/test splits for different datasets as well as different combinations (2_convert_to_our_format/2_split_train_test_pickle.py)
