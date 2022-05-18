import sys
sys.path.append('python-main/')
import ai_engine as ai
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process input regarding dataset')
	parser.add_argument("source", help="Folder containing images")
	parser.add_argument("--labelling", help="Utilizes automatic labelling", action="store_true")
	parser.add_argument("--split_ratio", type=float, help="The ratio in which the labeled data should be split in")
	parser.add_argument("--combine_bbs", help="Combines all bounding boxes present in automatic labeling", action="store_true")
	parser.add_argument("--combine", help="Artificially generates new combined data", action="store_true")
	parser.add_argument("--nr_components", type=int, help="The number of components supposed to be combined")
	parser.add_argument("--start_index", type=int, help="The index which images are suppose to start on")
	parser.add_argument("--component_folder", help="The folder in which components are located in if they arent present in source")

	args = parser.parse_args()
	
	if args.labelling:
		f = open('./datasets/labels.txt', "r")
		data = f.read().split('\n')
		f.close()
		labels = {data[i]:i for i in range(len(data))}
		root_folder = args.source
		
		ratio = 1
		split = True
		if args.split_ratio:
			ratio = args.split_ratio
			
		combine_bbs = False
		if args.combine_bbs:
			combine_bbs = True
			
		print('Generating labels for "'+root_folder+'"')
		ai.generate_dataset(root_folder, labels, split_components=split, train_val_ratio=ratio, combine_all=combine_bbs)
		print('Dataset has been generated at: '+root_folder+'_generated')
	elif args.combine:
		root_folder = args.source
		if not args.nr_components:
			print("Error: Number of components to combined is required, use --nr_components to set number of components")
			exit()
			
		start_index = 0
		if args.start_index:
			start_index = args.start_index
		
		print('Artificially combining images for dataset in "'+root_folder+'"')
		if args.component_folder:
			ai.generate_combined_data(root_folder, max_nr_components=args.nr_components, start_index_filename=start_index, component_folder=args.component_folder)
		else:
			ai.generate_combined_data(root_folder, max_nr_components=args.nr_components, start_index_filename=start_index)
		print('Dataset has been generated at: '+root_folder+'_combined_'+str(args.nr_components)+'')
		
		
