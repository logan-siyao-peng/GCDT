import os, io, re
import argparse
from glob import glob
import xmltodict

import stanza
from stanza.utils.conll import CoNLL

from utils import read_text_file, write_lines_file, get_basename_and_branch, get_file_modified_time


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


def write_parsed_file(output_parse, text_lines, meta_header, basename, parsed_file):
	assert len(output_parse) == len(text_lines)
	with io.open(parsed_file, "w", encoding="utf8") as f_parsed:
		f_parsed.write(meta_header)
		for chunk_id in range(len(output_parse)):
			f_parsed.write("# sent_id = %s-%d\n# text = %s\n%s\n\n"
			               % (basename, chunk_id + 1, text_lines[chunk_id], output_parse[chunk_id]))


def add_metadata(basename, branch):
	raw_dir = "../data/raw/"
	raw_lines = io.open(raw_dir + branch + os.sep + basename + ".txt", "r", encoding="utf8").read().strip().split("\n")
	raw_header_line = raw_lines[0]
	xml_dict = xmltodict.parse(raw_header_line, encoding="utf-8")["text"]
	meta_header = "# newdoc id = %s\n# meta::dateCollected = %s\n# meta::dateCreated = %s\n# meta::dateModified = %s\n" \
	              "# meta::sourceURL = %s\n# meta::speakerCount = %s\n# meta::title = %s\n" \
	              "# meta::shortTitle = %s\n# meta::author = %s\n# meta::genre = %s\n# meta::partition = %s\n" \
	              % (basename, xml_dict["@dateCollected"], xml_dict["@dateCreated"], xml_dict["@dateModified"],
	                 xml_dict["@sourceURL"], xml_dict["@speakerCount"], xml_dict["@title"],
	                 xml_dict["@shortTitle"], xml_dict["@author"], xml_dict["@genre"], branch)
	text_lines = [x for x in raw_lines if x.strip()!="" and not x.startswith("<")]
	return meta_header, text_lines
	

def parse_documents(parsed_output_dir, use_onto_trained=True, replace_existing = False):
	# Set up nlp
	nlp = set_up_nlp(has_gold_tokens=True, has_gold_sentences=True, use_onto_trained=use_onto_trained)
	input_dir = "../data/tokenized/"
	
	# Iterate through documents
	docs = sorted(list(glob(input_dir + "**/gcdt_*.txt", recursive=False)))
	for doc in docs:
		basename, branch = get_basename_and_branch(doc)
		parsed_doc = parsed_output_dir + branch + os.sep + basename + ".conllu"
		
		# if parsed_doc is newer, skip the current document
		if os.path.exists(parsed_doc):
			parsed_modified_time = get_file_modified_time(parsed_doc)
			doc_modified_time = get_file_modified_time(doc)
			if parsed_modified_time > doc_modified_time and not replace_existing:
				print("o Skipping %s:%s since parsed conllu file (%f) is newer than document (%f)"
			      % (branch, basename, parsed_modified_time,doc_modified_time))
				continue

		# parse document
		lines = read_text_file(doc, include_xml=False)
		tokens_list = []
		conllu_list = []
		for line in lines:
			tokens, conllu = parse_sentence(nlp, line)
			tokens_list.append(tokens)
			conllu_list.append(conllu)
		
		# Write or assert tokens
		mismatches = [(idx, lines[idx], tokens_list[idx]) for idx in range(len(tokens_list)) if lines[idx]!=tokens_list[idx]]
		assert tokens_list == lines
		
		# add metadata into conllu
		meta_header, text_lines = add_metadata(basename, branch)
		
		# Write parsed dependencies
		write_parsed_file(conllu_list, text_lines, meta_header, basename, parsed_doc)
		
		print("o Done with parsing %s:%s" % (branch, basename))
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--parsed_dir", default="../data/conllu/")
	parser.add_argument("--replace_existing", choices=[True, False], default=False)
	parser.add_argument("--use_onto_trained", choices=[True, False], default=True)
	args = parser.parse_args()
	
	# prepare directory and file
	division_dirs = [args.parsed_dir + "train/", args.parsed_dir + "dev/", args.parsed_dir + "test/"]
	for division_dir in division_dirs:
		if not os.path.isdir(division_dir):
			os.makedirs(division_dir)
		
	parse_documents(parsed_output_dir=args.parsed_dir,
                    use_onto_trained=args.use_onto_trained,
                    replace_existing = args.replace_existing)
	
	print('o ALL DONE!')