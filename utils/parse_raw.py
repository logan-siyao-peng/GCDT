import os, io, re
import argparse
from glob import glob

import stanza
from stanza.utils.conll import CoNLL


nlp_onto = stanza.Pipeline("zh", processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=False,
                           tokenize_no_ssplit=True,
                           pos_pretrain_path="../../../code/stanza-train/stanza/saved_models/pos/zh_ontonotes.pretrain.pt",
                           pos_model_path="../../../code/stanza-train/stanza/saved_models/pos/zh_ontonotes_tagger.pt",
                           depparse_pretrain_path="../../../code/stanza-train/stanza/saved_models/depparse/zh_ontonotes.pretrain.pt",
                           depparse_model_path="../../../code/stanza-train/stanza/saved_models/depparse/zh_ontonotes_parser.pt")

def parse_sentence(nlp, sentence):
	doc = nlp(sentence)  # Run the pipeline on input text
	dicts = doc.to_dict()
	conllu = CoNLL.convert_dict(dicts)
	tokens = " ".join([x[1] for x in conllu[0]])
	conllu = "\n".join(['\t'.join(x) for x in conllu[0]])
	return tokens, conllu


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--raw_dir", default="../data/raw/")
	parser.add_argument("--tokenized_dir", default="../data/tokenized/stanza/")
	parser.add_argument("--parsed_dir", default="../data/parsed/stanza/")
	args = parser.parse_args()
	
	for check_dir in [args.tokenized_dir, args.parsed_dir]:
		if not os.path.isdir(check_dir):
			os.makedirs(check_dir)
	
	raw_docs = sorted(list(glob(args.raw_dir + "gum_zh_*.txt", recursive=False)))
	for raw_doc in raw_docs:
		basename = os.path.splitext(os.path.basename(raw_doc))[0]
		tokenized_doc = args.tokenized_dir + basename + ".txt"
		parsed_doc = args.parsed_dir + basename + ".conllu"
		
		if os.path.exists(tokenized_doc) and os.path.exists(parsed_doc): continue
		
		with io.open(raw_doc, "r", encoding="utf8") as f_raw:
			raw_lines = f_raw.read().split("\n")
			
		tokens_list = []
		conllu_list = []
		for raw_line in raw_lines:
			if raw_line.startswith("<") or re.match(r"^\s*$", raw_line):
				continue
			tokens, conllu = parse_sentence(nlp_onto, raw_line)
			tokens_list.append(tokens)
			conllu_list.append(conllu)

		with io.open(tokenized_doc, "w", encoding="utf8") as f_tokenized, \
			io.open(parsed_doc, "w", encoding="utf8") as f_parsed:
			f_tokenized.write("\n".join(tokens_list)+"\n")
			f_parsed.write("\n\n".join(conllu_list)+"\n\n")

		print("o Done with parsing: ", basename)
	
	