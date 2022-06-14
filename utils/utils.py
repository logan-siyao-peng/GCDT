import io, re, os

def read_text_file(text_file, include_xml=True):
	"""
	Include .txt, .edu files (raw, token, and edu)
	:param text_file:
	:return:
	"""
	with io.open(text_file, "r", encoding="utf8") as f_text:
		text_lines = f_text.read().strip().split("\n")
	# Remove empty lines
	text_lines = [x for x in text_lines if not re.match(r"^\s*$", x)]
	if not include_xml:
		text_lines = [x for x in text_lines if not x.strip().startswith("<")]
	return text_lines
	

def write_parsed_file(output_parse, parsed_file):
	with io.open(parsed_file, "w", encoding="utf8") as f_parsed:
		f_parsed.write("\n\n".join(output_parse) + "\n\n")
		
def write_lines_file(output_lines, lines_file):
	"""
	Include .txt, .edu files (raw, tokenized, and edu)
	:param output_text:
	:param text_file:
	:return:
	"""
	with io.open(lines_file, "w", encoding="utf8") as f_lines:
		f_lines.write("\n".join(output_lines) + "\n")

def get_basename(filepath):
	return os.path.splitext(os.path.basename(filepath))[0]

def string_no_space(string):
	return re.sub(r'\s+', '', string)

def get_file_modified_time(filepath):
	return os.path.getctime(filepath)

def get_rel_name(line):
	relname = re.findall(r"relname=\"([^\"]+)\"", line)
	assert len(relname) <= 1
	return None if len(relname)==0 else relname[0]

def substring_of_some_item(s, l):
	return any(s in x for x in l)