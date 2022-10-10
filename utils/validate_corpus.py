import io, re, os
import argparse
import lxml
import xmlschema

from glob import glob
from collections import Counter, defaultdict

from utils import get_basename_and_branch, read_text_file, string_no_space, get_rel_name, substring_of_some_item

dev_filenames = ["gcdt_academic_peoples", "gcdt_bio_byron", "gcdt_interview_ward", "gcdt_news_famine", "gcdt_whow_hiking"]
test_filenames = ["gcdt_academic_dingzhen", "gcdt_bio_dvorak", "gcdt_interview_wimax", "gcdt_news_simplified", "gcdt_whow_thanksgiving"]


def main_validator():
	# Train/Dev/Test partition validators
	partition_validator()
	
	rs3_files = sorted(glob(args.data_dir + "rs3/**/*.rs3"))
	doc_stats_list = []
	
	prev_genre = ""
	for rs3_file in rs3_files:
		# Get basename and corresponding file names
		basename, branch = get_basename_and_branch(rs3_file)
		curr_genre = basename.split("_")[-2]
		
		if prev_genre != curr_genre:
			# print("\n\nStart genre: ", curr_genre)
			prev_genre = curr_genre
		
		
		# Step 1: Validate consistent tokenization
		# print("o Step 1: Validating consistent tokenization")
		tokenization_validator(basename, branch)
		
		# Step 2: print out docs statistics -- number of tokens, number of edus, and save relation counter
		# print("o Step 2: Print out doc stats")
		if branch != "double":
			doc_stats = get_doc_stats(basename, branch)
			doc_stats_list.append(doc_stats)

	# Step 999: print out corpus stats
	print("o There are a total of %d documents!" % len(doc_stats_list))
	total_num_tokens = sum([x[0] for x in doc_stats_list])
	total_num_edus = sum([x[1] for x in doc_stats_list])
	total_relation_counter = sum(map(Counter, [x[2] for x in doc_stats_list]), Counter())
	print("| %s | %d | %d |" % ("all_docs", total_num_tokens, total_num_edus))
	
	total_relation_items =sorted(total_relation_counter.items(), key=lambda x: x[1], reverse=True)
	total_num_relations = sum([x[1] for x in total_relation_items])
	print("\n\nTotal number of relations: ", total_num_relations)
	for k,v in total_relation_items:
		print("| %s | %d | %.2f |" % (k, v, 100.0*v/total_num_relations))
	
	print("o All validations are successful!")
	
def partition_validator():
	dirs = os.listdir(args.data_dir)
	for dir in dirs:
		if dir in ['others', '.DS_Store', 'README.md']:
			continue
		subdirs = os.listdir(args.data_dir + dir + os.sep)
		subdirs = [x for x in subdirs if not x.startswith(".")] # remove ".DS_Store" and/or other hidden files
		if dir in ["xml", "tokenized", "conllu"]:
			assert sorted(subdirs) == ["dev", "test", "train"]
		else:
			assert sorted(subdirs) == ["dev", "double", "test", "train"]
		for subdir in subdirs:
			files = os.listdir(args.data_dir + dir + os.sep + subdir + os.sep)
			files = [os.path.splitext(x)[0] for x in files if not x.startswith(".")] # remove ".DS_Store" and/or other hidden files
			files = sorted(files) # sort filenames
			if subdir == "dev":
				assert files == dev_filenames
			elif subdir in ["test", "double"]:
				assert files == test_filenames
			elif subdir == "train":
				assert len(files) == 40
				for file in files:
					assert file not in dev_filenames + test_filenames
	
	print("o Train/Dev/Test Partition Validation Completed!")

def xml_validator(xml_lines, xml_file):

	xml_stack = []
	for xml_line in xml_lines:
		if re.match(r"^<.+/>$", xml_line.strip()):
			continue
		elif not xml_line.startswith("<"):
			continue
		elif re.match(r"^<[^/].*[^/]>$", xml_line.strip()):
			xml_tag = re.findall(r"^<\S+[^ >]", xml_line.strip())
			assert len(xml_tag) == 1
			xml_tag = xml_tag[0][1:]
			xml_stack.append(xml_tag)
		elif re.match(r"^</[^/]+>$", xml_line.strip()):
			xml_tag = re.findall(r"^</\S+[^ >]", xml_line.strip())
			assert len(xml_tag) == 1
			xml_tag = xml_tag[0][2:]
			assert xml_stack[-1] == xml_tag
			xml_stack.pop(-1)
		else:
			assert False
	assert xml_stack == []
	
	return
			

def tokenization_validator(basename, branch):
	edu_file = args.data_dir + "rs3_extracted_edus" + os.sep + branch + os.sep + basename + ".edus"
	if branch == "double":
		token_file_branch = "test"
	else:
		token_file_branch = branch
	token_file = args.data_dir + "tokenized" + os.sep + token_file_branch + os.sep  + basename + ".txt"
	xml_file = args.data_dir + "xml" + os.sep + token_file_branch + os.sep  + basename + ".xml"
	edu_lines = read_text_file(edu_file, include_xml=True)
	token_lines = read_text_file(token_file, include_xml=True)
	xml_lines = read_text_file(xml_file, include_xml=True)
	
	# Validate xml in xml_lines
	xml_validator(xml_lines, xml_file)
	
	# remove xml in xml
	xml_lines = [x for x in xml_lines if not x.strip().startswith("<")]
	
	# Assert same no_space string across edu, token, xml files
	edu_lines_no_space = string_no_space("".join(edu_lines))
	token_lines_no_space = string_no_space("".join(token_lines))
	xml_lines_no_space = string_no_space("".join(xml_lines))
	
	assert  edu_lines_no_space == token_lines_no_space
	assert edu_lines_no_space ==  xml_lines_no_space
	
	# Assert no_space token line always contained in a no_space xml line
	for token_line in token_lines:
		assert token_line.strip() == token_line
		token_line_no_space = string_no_space(token_line)
		assert substring_of_some_item(token_line_no_space, [string_no_space(x) for x in xml_lines])
	
	# Assert edu line always contained in a token line (EDU never bigger than a sentence)
	for edu_line in edu_lines:
		assert edu_line.strip() == edu_line
		assert substring_of_some_item (edu_line, token_lines)
	
	return
	
def get_doc_stats(basename, branch):
	"""
	When given the basename, returns:
	- number of tokens and edus (based on edu file)
	- and the relation counter (based on rs3 file)
	:param basename:
	:return:
	"""
	edu_file = args.data_dir + "rs3_extracted_edus" + os.sep + branch + os.sep  + basename + ".edus"
	rs3_file = args.data_dir + "rs3" + os.sep  + branch + os.sep  + basename + ".rs3"
	edu_lines = read_text_file(edu_file, include_xml=True)
	rs3_lines = read_text_file(rs3_file, include_xml=True)
	
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
		
	return (num_tokens, num_edus, relation_counter)

	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default="../data/")
	args = parser.parse_args()
	
	main_validator()

	

	
		
		
		
		
		
		
		

