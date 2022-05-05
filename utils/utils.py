import io, re, os

def read_text_file(text_file):
	"""
	Include .txt, .edu files (raw, tokenized, and edu)
	:param text_file:
	:return:
	"""
	with io.open(text_file, "r", encoding="utf8") as f_text:
		text_lines = f_text.read().strip().split("\n")
	text_lines = [x for x in text_lines if not x.startswith("<") or not re.match(r"^\s*$", x)]
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


def read_lines_file(lines_file):
	"""
	Code partially borrowed from: https://github.com/amir-zeldes/gum/blob/dev/_build/utils/rst2dis.py
	:param rs3_file:
	:return:
	"""
	with io.open(lines_file, "r", encoding="utf8") as f_lines:
		input_lines = f_lines.read().strip().split("\n")
	return input_lines

def get_basename(filepath):
	return os.path.splitext(os.path.basename(filepath))[0]

def string_no_space(string):
	return re.sub(r'\s+', '', string)

def get_file_modified_time(filepath):
	return os.path.getctime(filepath)