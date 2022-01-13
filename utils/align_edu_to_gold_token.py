import io, re, os
import argparse
from glob import glob

from utils import get_basename, write_text_file, read_text_file, string_no_space

def align_edus_to_gold_tokens(edu_file, gold_token_file):
	edus = read_text_file(edu_file)
	gold_tokens = read_text_file(gold_token_file)
	
	aligned_edus = []
	gold_sid = 0
	gold_charid = 0
	for edu in edus:
		start_charid = gold_charid
		edu_no_space = string_no_space(edu)
		edu_no_space_charid = 0
		while edu_no_space_charid < len(edu_no_space):
			if gold_tokens[gold_sid][gold_charid] == " ":
				gold_charid += 1
			else:
				assert gold_tokens[gold_sid][gold_charid] == edu_no_space[edu_no_space_charid]
				edu_no_space_charid += 1
				gold_charid += 1
		
		assert gold_charid == len(gold_tokens[gold_sid]) or gold_tokens[gold_sid][gold_charid] == " "
		aligned_edus.append(gold_tokens[gold_sid][start_charid:gold_charid].strip())
		
		if gold_charid == len(gold_tokens[gold_sid]):
			gold_sid += 1
			gold_charid = 0
	
	if edus == aligned_edus:
		print("o Edu file is already gold tokenized, no need to align: ", edu_file)
	else:
		aligned_edu_file = edu_file + ".aligned"
		write_text_file(aligned_edus, aligned_edu_file)
		print('o Done with aligning: ', aligned_edu_file)
		
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--gold_token_dir", default="../data/tokenized/gold/")
	parser.add_argument("--edu_dir", default="../data/edu/")
	parser.add_argument("--annotator", choices=["logan", "janet"], default="janet")
	args = parser.parse_args()
	
	edu_files = sorted(glob(args.edu_dir + args.annotator + os.sep + "*.edu"))
	
	for edu_file in edu_files:
		basename = get_basename(edu_file)
		gold_token_file = args.gold_token_dir + basename + ".txt"
		align_edus_to_gold_tokens(edu_file, gold_token_file)
	
	print("o ALL DONE!")
		
		
		
		
		
		
		
		

