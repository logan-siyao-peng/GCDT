import io, re, os
import argparse
from glob import glob
from deep_translator import GoogleTranslator
translator = GoogleTranslator(source='zh-CN', target='en')

from utils import read_lines_file, get_basename, write_lines_file

def auto_trans_rs3(rs3_file, autotran_rs3_dir=None):
	rs3_lines = read_lines_file(rs3_file)
	edus = []
	for line_id, line in enumerate(rs3_lines):
		if "<segment" in line:
			m = re.search(r'<segment id="([^"]+)"[^>]*>(.*?)</segment>', line)
			if m is not None:
				m_text = m.group(2).strip()
				if re.match(r"^\d+$", m_text):
					m_trans = m_text
				else:
					m_trans = translator.translate(m_text)
				rs3_lines[line_id] = rs3_lines[line_id].replace(m_text, "%s // %s" % (m_trans, m_text))
				print("o Done with translating line: %d" % line_id, end="\r")
	
	# Write edus if edu_dir exists
	if autotran_rs3_dir:
		basename = get_basename(rs3_file)
		autotran_rs3_file = autotran_rs3_dir + basename + ".rs3"
		if not os.path.isdir(autotran_rs3_dir):
			os.makedirs(autotran_rs3_dir)
		write_lines_file(rs3_lines, autotran_rs3_file)
		print("o Done autotranslated rs3 file: ", basename)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--rs3_dir", default="../data/rs3/")
	parser.add_argument("--autotran_rs3_dir", default="../data/autotran_rs3/")
	args = parser.parse_args()
	
	# rs3_files = sorted(glob(args.rs3_dir + "*.rs3", recursive=False))
	rs3_files = [
	             "/Users/loganpeng/Dropbox/Dissertation/data/GUM_Chinese/data/rs3/gum_zh_whow_quinoa.rs3",
	]
	for file in rs3_files:
		auto_trans_rs3(file, autotran_rs3_dir=args.autotran_rs3_dir)

