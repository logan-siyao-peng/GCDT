import io, re, os
from glob import glob
import sklearn
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, accuracy_score
import argparse
from functools import reduce


def read_edu_to_binaries(filename):
	with io.open(filename, "r", encoding="utf8") as f:
		lines = f.read().strip().split("\n")
	binaries = [[0] * x.strip().count(" ") + [1] for x in lines]
	binaries = reduce(lambda x,y: x+y, binaries)
	return binaries, lines

def segmentation_agreement(y1, y2):
	cohen_kappa_result = cohen_kappa_score(y1, y2)
	confusion_matrix_result = confusion_matrix(y1, y2)
	f1_score_result = f1_score(y1, y2)
	raw_accuracy_result = accuracy_score(y1, y2)
	print("Cohen's kappa:	", cohen_kappa_result)
	print("Confusion matrix:    ", confusion_matrix_result)
	print("F1 score:    ", f1_score_result)
	print("Raw accuracy:    ", raw_accuracy_result)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--janet_dir", "-j", default="../data/JanetIAA/segmented/")
	parser.add_argument("--logan_dir", "-l", default="../data/rs3_extracted_edu/")
	args = parser.parse_args()
	
	iaa_docs = [os.path.basename(x) for x in sorted(glob(args.janet_dir + "*.txt"))]
	for iaa_doc in iaa_docs:
		print("\nDocument: ", iaa_doc)
		janet_binaries, janet_lines = read_edu_to_binaries(args.janet_dir + iaa_doc)
		logan_binaries, logan_lines = read_edu_to_binaries(args.logan_dir + iaa_doc.replace(".txt", ".edu"))
		segmentation_agreement(janet_binaries, logan_binaries)
	
