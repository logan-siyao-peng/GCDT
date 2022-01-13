import io, re, os
import argparse
from glob import glob

from utils import read_rs3_file, get_basename, write_text_file

def extract_edus_from_rs3(rs3_file, edu_dir=None):
	rs3_lines = read_rs3_file(rs3_file)
	edus = []
	for line in rs3_lines:
		if "<segment" in line:
			m = re.search(r'<segment id="([^"]+)" >(.*?)</segment>', line) # without relation annotation
			# m = re.search(r'<segment id="([^"]+)" parent="([^"]+)" relname="([^"]+)">(.*?)</segment>', line) # with relation annotation
			if m is not None:
				edus.append((int(m.group(1)), m.group(2).strip()))
	edus = [x[1] for x in sorted(edus, key=lambda x: x[0])]
	
	# Write edus if edu_dir exists
	if edu_dir:
		basename = get_basename(rs3_file)
		edu_file = edu_dir + basename + ".edu"
		if not os.path.isdir(edu_dir):
			os.makedirs(edu_dir)
		write_text_file(edus, edu_file)
		print("o Done extracting file: ", basename)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--rs3_dir", default="../data/rs3/tmp/")
	parser.add_argument("--edu_dir", default="../data/edu/")
	args = parser.parse_args()
	
	rs3_files = sorted(glob(args.rs3_dir + "*.rs3", recursive=False))
	for file in rs3_files:
		extract_edus_from_rs3(file, edu_dir=args.edu_dir)
	
	