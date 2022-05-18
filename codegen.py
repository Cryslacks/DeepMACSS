import sys
sys.path.append('python-main/')
sys.path.append('language_generators/')
import codegen_engine as cg
import argparse
import json

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process input regarding dataset')
	parser.add_argument("source", help="Location of file containing the semantics to be generated from")
	parser.add_argument("language", help="Which language generator to utilize to generate code")
	parser.add_argument("destination", help="Location which the generated code to be saved")

	args = parser.parse_args()
	
	f = open(args.source, "r")
	data = f.read()
	f.close()
	
	cg.generate_code(data, args.language, args.destination)   