import sys
sys.path.append('python-main/')
import ai_engine as ai
import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process input regarding dataset')
	
	parser.add_argument("--train", help="Trains a model on a specified dataset", action="store_true")
	parser.add_argument("--train_extend", help="Retrains a model for a new component", action="store_true")
	parser.add_argument("--predict", help="Utilizes the model to make a prediction on an image", action="store_true")
	parser.add_argument("--evaluate", help="Evaluates a model on a defined dataset", action="store_true")
	parser.add_argument("--image", help="Image to predict on")
	parser.add_argument("--iou", type=float, help="IoU used when predicting")
	parser.add_argument("--min_score", type=float, help="Minimum score the model to have it account for in its prediction")
	parser.add_argument("--min_text_score", type=float, help="Minimum score for the OCR to have it account for in its prediction")
	parser.add_argument("--destination", help="Defines the destination location of the generated semantics")
	parser.add_argument("--dataset", help="Dataset to process")
	parser.add_argument("--combined", help="Specifies if the dataset contains artificially combined images", action="store_true")
	parser.add_argument("--model_date", help="Utilized to load a predefined model")
	parser.add_argument("--model_name", help="Utilized to load a predefined model")
	parser.add_argument("--new_model_name", help="Utilized to save a newly trained model")
	parser.add_argument("--epochs", type=int, help="Defines the number of epochs to train the model for")
	parser.add_argument("--eval_sub_folder", help="Defines the sub folder for dataset used in evaluation")
	parser.add_argument("--num_classes", type=int, help="Defines the number of classes contained within a dataset, used in evaluation")
	parser.add_argument("--map_percentage", type=float, help="Defines the precentage in which the mAP value is to be obtained for")

	args = parser.parse_args()

	if args.train:
		combined = False
		if args.combined:
			combined = True
		
		if not args.dataset:
			print('Error: No dataset provided, provide one via --dataset')
			exit()
			
		dataset_train = ai.SketchDataset('./'+args.dataset, 'train', combined=combined, preprocessed=combined)
		dataset_val = ai.SketchDataset('./'+args.dataset, 'val', combined=combined, preprocessed=combined)

		data_loader = torch.utils.data.DataLoader(
				dataset_train, batch_size=3, shuffle=True, num_workers=0,
				collate_fn=ai.collate_fn)
		data_loader_val = torch.utils.data.DataLoader(
				dataset_val, batch_size=3, shuffle=False, num_workers=0,
				collate_fn=ai.collate_fn)

		device = torch.device('cpu') #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		if args.model_date and args.model_name:
			model,optimizer = ai.load_frcnn(args.model_date, args.model_name)
		else:
			model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
			
			if not args.num_classes:
				args.num_classes = 13
			num_classes = args.num_classes
			in_features = model.roi_heads.box_predictor.cls_score.in_features
			model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
			
			params = [p for p in model.parameters() if p.requires_grad]
			optimizer = torch.optim.SGD(params, lr=0.005, 
										momentum=0.9, weight_decay=0.0005)
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
													   step_size=3,
													   gamma=0.1)
		model.train()
		if not args.new_model_name:
			print('Error: Model needs to have a name, set one via --new_model_name')
			exit()
		if not args.epochs:
			print('Error: Epochs needs to be specified, set the number via --epochs')
			exit()
		ai.train_model(model, optimizer, data_loader, data_loader_val, device, args.epochs, 'Faster-RCNN', args.new_model_name, lr_scheduler)
		
	elif args.train_extend:
		combined = False
		if args.combined:
			combined = True
			
		if not args.dataset:
			print('Error: No dataset provided, provide one via --dataset')
			exit()
			
		dataset_train = ai.SketchDataset('./'+args.dataset, 'train', combined=combined, preprocessed=combined)
		dataset_val = ai.SketchDataset('./'+args.dataset, 'val', combined=combined, preprocessed=combined)

		data_loader = torch.utils.data.DataLoader(
				dataset_train, batch_size=3, shuffle=True, num_workers=0,
				collate_fn=ai.collate_fn)
		data_loader_val = torch.utils.data.DataLoader(
				dataset_val, batch_size=3, shuffle=False, num_workers=0,
				collate_fn=ai.collate_fn)

		device = torch.device('cpu') #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

		if not args.model_date or not args.model_name:
			print('Error: It is required to specifiy a existing model to train a new component, use --model_date and --model_name to define it') 
			exit()
		
		model,_ = ai.load_frcnn(args.model_date, args.model_name)

		# Increase the number of outputs
		num_classes = model.roi_heads.box_predictor.cls_score.out_features + 1
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

		params = [p for p in model.parameters() if p.requires_grad]
		optimizer = torch.optim.SGD(params, lr=0.005, 
									momentum=0.9, weight_decay=0.0005)
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
													   step_size=3,
													   gamma=0.1)
		model.train()
		
		if not args.new_model_name:
			print('Error: Model needs to have a name, set one via --new_model_name')
			exit()
		if not args.epochs:
			print('Error: Epochs needs to be specified, set the number via --epochs')
			exit()
		ai.train_model(model, optimizer, data_loader, data_loader_val, device, args.epochs, 'Faster-RCNN', args.new_model_name, lr_scheduler)
	elif args.predict:
		if not args.image:
			print("Error: No image specified, specify one by using --image")
			exit()
		
		if not args.model_date or not args.model_name:
			print('Error: It is required to specifiy a existing model to train a new component, use --model_date and --model_name to define it') 
			exit()
		
		model,_ = ai.load_frcnn(args.model_date, args.model_name)
		
		f = open('./datasets/labels.txt', "r")
		data = f.read().split('\n')
		f.close()
		labels_to_idx = {data[i]:i for i in range(len(data))}
		idx_to_labels = [data[i] for i in range(len(data))]
		
		iou = 0.5
		if args.iou:
			iou = args.iou
		
		min_pred = 0.8
		if args.min_score:
			min_pred = args.min_score
		
		min_text_pred = 0.4
		if args.min_text_score:
			min_text_pred = args.min_text_score
		
		
		prediction = ai.predict_model(model, args.image, IoU=iou, disregard_comp=[10,11,12], priority_comp=[11])
		txt_prediction = ai.predict_text(args.image)
		comp_defs = ai.get_component_definitions('datasets/definitions.json', labels_to_idx)

		json_str = ai.post_process_results((prediction, txt_prediction), comp_defs, idx_to_labels, min_score=[min_pred, min_text_pred])
		if args.destination:
			f = open(args.destination, "w")
			f.write(json_str)
			f.close()
		else:
			print('\n'+json_str+'\n')
		
		
	elif args.evaluate:
		combined = False
		if args.combined:
			combined = True
		
		if not args.dataset:
			print('Error: No dataset provided, provide one via --dataset')
			exit()
			
		sub_folder = ''
		if args.eval_sub_folder:
			sub_folder = args.eval_sub_folder
		dataset_train = ai.SketchDataset('./'+args.dataset, sub_folder, combined=combined, preprocessed=combined)
		data_loader_map = torch.utils.data.DataLoader(
				dataset_map, batch_size=3, shuffle=False, num_workers=0,
				collate_fn=ai.collate_fn)

		if not args.model_date or not args.model_name:
			print('Error: It is required to specifiy a existing model to train a new component, use --model_date and --model_name to define it') 
			exit()
			
		if not args.num_classes:
			args.num_classes = 13
		model,_ = load_frcnn(args.model_date, args.model_name, num_classes=args.num_classes)
		model.eval()
	
		percentage = 0.5
		if args.map_percentage:
			percentage = args.map_percentage
		ai.calc_multi_map(model, data_loader_map, percentages=[percentage], print_res=True)
		