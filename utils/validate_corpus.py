import io, re, os
import argparse
from glob import glob
from collections import Counter, defaultdict

from utils import get_basename, read_text_file, string_no_space, get_rel_name, substring_of_some_item


def main_validator():
	rs3_files = sorted(glob(args.rs3_dir + os.sep + "*.rs3"))
	doc_stats_list = []
	
	for rs3_file in rs3_files:
		# Get basename and corresponding file names
		basename = get_basename(rs3_file)
		
		# print("o Start validating file: ", basename)
		
		# Step 1: Validate consistent tokenization
		# print("o Step 1: Validating consistent tokenization")
		tokenization_validator(basename)
		
		# Step 2: print out docs statistics -- number of tokens, number of edus, and save relation counter
		# print("o Step 1: Validating consistent tokenization")
		doc_stats = get_doc_stats(basename)
		doc_stats_list.append(doc_stats)
		
		# TODO: other validations
	
		# print("o Step 998: Ending validation for file: ", basename)

	# Step 999: print out corpus stats
	total_num_tokens = sum([x[0] for x in doc_stats_list])
	total_num_edus = sum([x[1] for x in doc_stats_list])
	total_relation_counter = sum(map(Counter, [x[2] for x in doc_stats_list]), Counter())
	print("%s\t%d\t%d" % ("all_docs", total_num_tokens, total_num_edus))
	
	total_relation_items =sorted(total_relation_counter.items(), key=lambda x: x[1], reverse=True)
	total_num_relations = sum([x[1] for x in total_relation_items])
	print("\n\nTotal number of relations: ", total_num_relations)
	for k,v in total_relation_items:
		print("%s\t%d\t%.2f" % (k, v, 100.0*v/total_num_relations))
	
	print("o All validations are successful!")


def tokenization_validator(basename):
	edu_file = args.edu_dir + basename + ".edu"
	token_file = args.token_dir + basename + ".txt"
	raw_file = args.raw_dir + basename + ".txt"
	edu_lines = read_text_file(edu_file)
	token_lines = read_text_file(token_file)
	raw_lines = read_text_file(raw_file)
	
	# Assert same no_space string across edu, token, raw files
	assert  string_no_space("".join(edu_lines)) == string_no_space("".join(token_lines))
	assert string_no_space("".join(edu_lines)) == string_no_space("".join(raw_lines))
	
	# Assert no_space token line always contained in a no_space raw line
	for token_line in token_lines:
		assert token_line.strip() == token_line
		assert substring_of_some_item(string_no_space(token_line), [string_no_space(x) for x in raw_lines])
	
	# Assert edu line always contained in a token line (EDU never bigger than a sentence)
	for edu_line in edu_lines:
		assert edu_line.strip() == edu_line
		assert substring_of_some_item (edu_line, token_lines)
	
	return
	
def get_doc_stats(basename):
	"""
	When given the basename, returns:
	- number of tokens and edus (based on edu file)
	- and the relation counter (based on rs3 file)
	:param basename:
	:return:
	"""
	edu_file = args.edu_dir + basename + ".edu"
	rs3_file = args.rs3_dir + basename + ".rs3"
	edu_lines = read_text_file(edu_file)
	rs3_lines = read_text_file(rs3_file)
	
	# Get number of tokens and edus
	num_tokens= " ".join(edu_lines).strip().count(" ")+1
	num_edus =  len(edu_lines)
	print("%s\t%d\t%d" % (basename, num_tokens, num_edus))
	
	# get relation counter
	relation_counter = defaultdict(int)
	for rs3_line in rs3_lines:
		relname = get_rel_name(rs3_line)
		if relname and relname not in ["span"]:
			relation_counter[relname] += 1
	
	# Make sure the number of relations is equal to number of edus - 1
	# TODO: ask Amir
	# assert sum(relation_counter.values()) == num_edus - 1
	
	return (num_tokens, num_edus, relation_counter)

	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--raw_dir", default="../data/raw/")
	parser.add_argument("--token_dir", default="../data/token/")
	parser.add_argument("--edu_dir", default="../data/rs3_extracted_edu/")
	parser.add_argument("--rs3_dir", default="../data/rs3/")
	args = parser.parse_args()
	
	main_validator()

	

	
		
		
		
		
		
		
		

