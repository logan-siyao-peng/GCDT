import io, re, os
import argparse
from glob import glob
from time import time
from deep_translator import GoogleTranslator


from utils import read_text_file, get_basename_and_branch, write_lines_file, get_file_modified_time

def auto_trans_rs3(rs3_file, autotran_rs3_dir):
	# prepare directory and file
	if not os.path.isdir(autotran_rs3_dir):
		os.makedirs(autotran_rs3_dir)
	basename, branch = get_basename_and_branch(rs3_file)
	autotran_rs3_file = autotran_rs3_dir + branch + os.sep + basename + ".rs3"
	
	# if the autotrans file is newer than the chinese-only rs3 file then don't translate this file
	if os.path.exists(autotran_rs3_file):
		autotran_modified_time = get_file_modified_time(autotran_rs3_file)
		original_modified_time = get_file_modified_time(rs3_file)
		if autotran_modified_time > original_modified_time:
			print("o Skipping %s:%s since translated file (%f) is newer than original (%f)"
		      % (branch, basename, autotran_modified_time,original_modified_time))
			return
	
	# read and translate rs3 lines
	print("o Start translating file: %s:%s" % (branch, basename))
	rs3_lines = read_text_file(rs3_file, include_xml=True)
	edus = []
	for line_id, line in enumerate(rs3_lines):
		if "<segment" in line:
			m = re.search(r'<segment id="([^"]+)"[^>]*>(.*?)</segment>', line)
			if m is not None:
				m_end = int(m.end(2))
				m_text = m.group(2).strip()
				# print(m_text)
				if re.match(r"^\d+$", m_text):
					# list of special cases here:
					# - do not translate numbers
					# - * * *
					m_trans = m_text
				else:
					m_trans = translator.translate(m_text)
					if m_trans == None:
						m_trans = m_text
				
				m_trans = m_trans.replace(" & ", " &amp; ").strip()
				
				# rs3_lines[line_id] = rs3_lines[line_id].replace(">"+m_text+"<", ">"+m_trans+"<") #Logan: only translation
				rs3_lines[line_id] = rs3_lines[line_id][:m_end] + " // " + m_trans + rs3_lines[line_id][m_end:] # both source and translations
				print("o Done with translating line: %d" % line_id, end="\r")
	
	# write autotrans_rs3_lines
	write_lines_file(rs3_lines, autotran_rs3_file)
	print("o Done autotranslated rs3 file %s:%s" % (branch, basename))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--source_dir",
	                    # default="../../../code/loganfolked_DMRST_Parser/data/rs3/en-gum-20220625/",
	                    default="../data/rs3/",
	                    )
	parser.add_argument("--target_dir",
	                    # default="../../../code/loganfolked_DMRST_Parser/data/rs3/entranszh-gum-20220705/",
	                    default="../data/autotrans_rs3/",
	                    )
	parser.add_argument("--source_language", default='en')
	parser.add_argument("--target_language", default='zh-CN')
	args = parser.parse_args()
	
	translator = GoogleTranslator(source=args.source_language, target=args.target_language)
	rs3_files = sorted(glob(args.source_dir + "**/*.rs3", recursive=False))
	print("We have in total %d documents to translate from %s to %s:"
	      % (len(rs3_files), args.source_language, args.target_language))
	
	# rs3_files = [
	#              "/Users/loganpeng/Dropbox/Dissertation/data/GUM_Chinese/data/rs3/gum_zh_whow_quinoa.rs3",
	# ]
	
	for file in rs3_files:
		auto_trans_rs3(file, autotran_rs3_dir=args.target_dir)

