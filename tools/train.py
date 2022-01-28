import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from vedastr.runners import TrainRunner  # noqa 402
from vedastr.utils import Config  # noqa 402

#import wandb




def parse_args():
	parser = argparse.ArgumentParser(description='Train.')
	parser.add_argument('--config', type=str, help='config file path')
	parser.add_argument('--distribute', default=False, action='store_true')
	parser.add_argument('--local_rank', type=int, default=0)
	parser.add_argument("--wandb",type=str, default=None)
	
	args = parser.parse_args()
	if 'LOCAL_RANK' not in os.environ:
		os.environ['LOCAL_RANK'] = str(args.local_rank)
	return args



def main():
	args = parse_args()
	
	cfg_path = args.config
	cfg = Config.fromfile(cfg_path)
	
	_, fullname = os.path.split(cfg_path)
	fname, ext = os.path.splitext(fullname)
	
	root_workdir = cfg.pop('root_workdir')
	workdir = os.path.join(root_workdir, fname)
	os.makedirs(workdir, exist_ok=True)
	# copy corresponding cfg to workdir
	shutil.copy(cfg_path, os.path.join(workdir, os.path.basename(cfg_path)))
	
	train_cfg = cfg['train']
	inference_cfg = cfg['inference']
	common_cfg = cfg['common']
	common_cfg['workdir'] = workdir
	common_cfg['distribute'] = args.distribute
	"""	if args.wandb:
		config={
			"learning_rate" : train_cfg["optimizer"]["lr"],
			"dropout" : cfg["dropout"],
			"optimizer" : train_cfg["optimizer"]["type"],
			"config" : os.path.basename(cfg_path),
			"augmentation_prob" : train_cfg["data"]["train"]["transform"][2]["prob"],
			"weight_decay" : train_cfg["optimizer"]["weight_decay"],
			"batch_size" : cfg["samples_per_gpu"],
			"num_mdcdp_layers" : cfg["num_mdcdp_layers"],
			"d_model" : cfg["d_model"],
		}
		wandb.init(project=args.wandb, entity="cs20m064")
		runner = TrainRunner(train_cfg, inference_cfg, common_cfg, wandb)
		
	else:"""
	print("wandb project name not passed in arguments")
	runner= TrainRunner(train_cfg, inference_cfg, common_cfg)
		
	runner()
	#wandb.finish()


if __name__ == '__main__':
	main()
