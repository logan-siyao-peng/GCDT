import io, re, os
import argparse
from glob import glob

from utils import read_text_file, get_basename, write_lines_file, get_file_modified_time

def extract_edus_from_rs3(rs3_file, extracted_edu_file=None):
	rs3_lines = read_text_file(rs3_file)
	edus = []
	for line in rs3_lines:
		if "<segment" in line:
			m = re.search(r'<segment id="([^"]+)"[^>]*>(.*?)</segment>', line)
			if m is not None:
				edus.append((int(m.group(1)), m.group(2).strip()))
	edus = [x[1].replace("&amp;", "&") for x in sorted(edus, key=lambda x: x[0])]
	write_lines_file(edus, extracted_edu_file)
	print("o Done extracting file: ", basename)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--lang", choices=["zh", "bi"], default="zh")
	args = parser.parse_args()
	
	if args.lang == "zh":
		rs3_dir = "../data/rs3/"
		extracted_edu_dir = "../data/rs3_extracted_edu/"
		
	elif args.lang == "bi":
		rs3_dir = "../data/autotrans_rs3/"
		extracted_edu_dir = "../data/autotrans_extracted_edu/"
	
	if not os.path.isdir(extracted_edu_dir):
		os.makedirs(extracted_edu_dir)
	
	print("\n\no Start Extracting EDUS for language model: ", args.lang, "\n")
	rs3_files = sorted(glob(rs3_dir + "*.rs3", recursive=False))
	
	for rs3_file in rs3_files:
		basename = get_basename(rs3_file)
		extracted_edu_file = extracted_edu_dir + basename + ".edu"
		
		# if the edu file is newer than the rs3 file then don't translate this file
		if os.path.exists(extracted_edu_file):
			extracted_edu_modified_time = get_file_modified_time(extracted_edu_file)
			original_modified_time = get_file_modified_time(rs3_file)
			if extracted_edu_modified_time > original_modified_time:
				# print("o Skipping %s since extracted edu file (%f) is newer than original (%f)"
				#       % (basename, extracted_edu_modified_time, original_modified_time))
				continue
		
		extract_edus_from_rs3(rs3_file, extracted_edu_file=extracted_edu_file)
	
	