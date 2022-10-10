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
	accuracy_result = accuracy_score(y1, y2)
	print("Cohen's kappa:	", cohen_kappa_result)
	print("Confusion matrix:    ", confusion_matrix_result)
	print("F1 score:    ", f1_score_result)
	print("Accuracy:    ", accuracy_result)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--double_dir", "-d", default="../data/others/double-seg-test/")
	parser.add_argument("--main_dir", "-m", default="../data/rs3_extracted_edus/test/")
	args = parser.parse_args()
	
	iaa_docs = [os.path.basename(x) for x in sorted(glob(args.double_dir + "*.edus"))]
	for iaa_doc in iaa_docs:
		print("\nDocument: ", iaa_doc)
		double_binaries, double_lines = read_edu_to_binaries(args.double_dir + iaa_doc)
		main_binaries, main_lines = read_edu_to_binaries(args.main_dir + iaa_doc)
		segmentation_agreement(double_binaries, main_binaries)
	
