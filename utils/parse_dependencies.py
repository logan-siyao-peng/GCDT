import os, io, re
import argparse
from glob import glob

import stanza
from stanza.utils.conll import CoNLL

from utils import read_text_file, write_parsed_file, write_lines_file, get_basename, get_file_modified_time


def set_up_nlp(has_gold_tokens=True, has_gold_sentences=True, use_onto_trained=True):
	model_path = {} if not use_onto_trained else {
		"pos_pretrain_path": "../../../code/stanza-train/stanza/saved_models/pos/zh_ontonotes.pretrain.pt",
        "pos_model_path" : "../../../code/stanza-train/stanza/saved_models/pos/zh_ontonotes_tagger.pt",
        "depparse_pretrain_path": "../../../code/stanza-train/stanza/saved_models/depparse/zh_ontonotes.pretrain.pt",
        "depparse_model_path": "../../../code/stanza-train/stanza/saved_models/depparse/zh_ontonotes_parser.pt"}
	nlp = stanza.Pipeline("zh", processors='tokenize,pos,lemma,depparse',
	                           tokenize_pretokenized=has_gold_tokens,
	                           tokenize_no_ssplit=has_gold_sentences,
	                           **model_path
	                           )
	return nlp

def parse_sentence(nlp, sentence):
	doc = nlp(sentence)  # Run the pipeline on input text
	dicts = doc.to_dict()
	conllu = CoNLL.convert_dict(dicts)
	tokens = " ".join([x[1] for x in conllu[0]])
	conllu = "\n".join(['\t'.join(x) for x in conllu[0]])
	return tokens, conllu

def parse_documents(parsed_output_dir, mode="token", use_onto_trained=True, replace_existing = False):
	# Set up nlp
	if mode == "token":
		nlp = set_up_nlp(has_gold_tokens=True, has_gold_sentences=True, use_onto_trained=use_onto_trained)
	elif mode == "raw":
		nlp = set_up_nlp(has_gold_tokens=False, has_gold_sentences=True, use_onto_trained=use_onto_trained)
	
	if mode == "raw":
		input_dir = "../data/raw/"
	elif mode == "token":
		input_dir = "../data/token/"
	
	# Set up output tokenized directory if parse from raw
	if mode == "raw":
		tokenized_output_dir = "../data/tokenized/stanza/"
		if not os.path.isdir(tokenized_output_dir):
			os.makedirs(tokenized_output_dir)

	# Iterate through documents
	docs = sorted(list(glob(input_dir + "gum_zh_*.txt", recursive=False)))
	for doc in docs:
		basename = get_basename(doc)
		parsed_doc = parsed_output_dir + basename + ".conllu"
		
		# if parsed_doc is newer, skip the current document
		if os.path.exists(parsed_doc):
			parsed_modified_time = get_file_modified_time(parsed_doc)
			doc_modified_time = get_file_modified_time(doc)
			if parsed_modified_time > doc_modified_time and not replace_existing:
				print("o Skipping %s since parsed conllu file (%f) is newer than document (%f)"
			      % (basename, parsed_modified_time,doc_modified_time))
				continue

		
		if mode == "raw":
			tokenized_doc = tokenized_output_dir + basename + ".txt"
		

		# parse document
		lines = read_text_file(doc)
		tokens_list = []
		conllu_list = []
		for line in lines:
			tokens, conllu = parse_sentence(nlp, line)
			tokens_list.append(tokens)
			conllu_list.append(conllu)
		
		# Write or assert tokens
		if mode == "raw":
			write_lines_file(tokens_list, tokenized_doc)
		elif mode == "token":
			mismatches = [(idx, lines[idx], tokens_list[idx]) for idx in range(len(tokens_list)) if lines[idx]!=tokens_list[idx]]
			assert tokens_list == lines
			
		# Write parsed dependencies
		write_parsed_file(conllu_list, parsed_doc)
		
		print("o Done with parsing: ", basename)
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=["raw", "token"], default="token")
	parser.add_argument("--parsed_dir", default="../data/parsed/stanza/")
	parser.add_argument("--replace_existing", choices=[True, False], default=False)
	parser.add_argument("--use_onto_trained", choices=[True, False], default=True)
	args = parser.parse_args()
	
	if not os.path.isdir(args.parsed_dir):
		os.makedirs(args.parsed_dir)
		
	parse_documents(parsed_output_dir=args.parsed_dir,
                    mode=args.mode,
                    use_onto_trained=args.use_onto_trained,
                    replace_existing = args.replace_existing)
	
	print('o ALL DONE!')