import argparse
import os
import sys

import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import cv2  # noqa 402
from vedastr.runners import InferenceRunner  # noqa 402
from vedastr.utils import Config  # noqa 402


def grid_sample_op(g, input1, input2, mode, padding_mode, align_corners):
	return g.op("torch::grid_sampler", input1, input2, mode, padding_mode, align_corners)


torch.onnx.register_custom_op_symbolic('::grid_sampler', grid_sample_op, 11)


def parse_args():
	parser = argparse.ArgumentParser(description='Inference')
	parser.add_argument('--config', type=str, help='config file path')
	parser.add_argument('--checkpoint', type=str, help='checkpoint file path')
	parser.add_argument('--image', type=str, help='sample image path')
	parser.add_argument('--out', type=str, help='output model file name')
	parser.add_argument(
		'--onnx', default=False, action='store_true', help='convert to onnx')
	parser.add_argument(
		'--max_batch_size',
		default=1,
		type=int,
		help='max batch size for trt engine execution')
	parser.add_argument(
		'--max_workspace_size',
		default=1,
		type=int,
		help='max workspace size for building trt engine')
	parser.add_argument(
		'--fp16',
		default=False,
		action='store_true',
		help='convert to trt engine with fp16 mode')
	parser.add_argument(
		'--int8',
		default=False,
		action='store_true',
		help='convert to trt engine with int8 mode')
	parser.add_argument(
		'--calibration_mode',
		default='entropy_2',
		type=str,
		choices=['entropy_2', 'entropy', 'minmax'])
	parser.add_argument(
		'--calibration_images',
		default=None,
		type=str,
		help='images dir used when int8 mode is True')
	args = parser.parse_args()
	
	return args


def main():
	args = parse_args()
	out_name = args.out
	
	cfg_path = args.config
	cfg = Config.fromfile(cfg_path)
	
	infer_cfg = cfg['inference']
	common_cfg = cfg.get('common')
	
	runner = InferenceRunner(infer_cfg, common_cfg)
	assert runner.use_gpu, 'Please use valid gpu to export model.'
	runner.load_checkpoint(args.checkpoint)
	
	image = cv2.imread(args.image)
	beam_size = 0
	aug = runner.transform(image=image, label='')
	image, label = aug['image'], aug['label']  # noqa 841
	image = image.unsqueeze(0).cuda()
	model = runner.model.cuda().eval()
	scripted_model = torch.jit.script(model)
	dummy_input = (image,)
	lang_id = torch.tensor([0]).cuda()
	
	if model.need_text :
		dummy_input +=   (runner.converter.test_encode([''])[0],)
	
	if model.need_lang:
		dummy_input += (lang_id,)
	
	dummy_input = list(dummy_input)
	torch_out = scripted_model(dummy_input)
	
	
	torch.onnx.export(scripted_model,  # model being run
          dummy_input,  # model input (or a tuple for multiple inputs)
          args.out,  # where to save the model (can be a file or file-like object)
          export_params=True,  # store the trained parameter weights inside the model file
          opset_version=11,  # the ONNX version to export the model to
          do_constant_folding=True,  # whether to execute constant folding for optimization
          input_names=['input', 'label', 'lang'],  # the model's input names
          output_names=['output'],  # the model's output names
          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                        'output': {0: 'batch_size'}})
		
	import onnx
	onnx_model = onnx.load(args.out)
	
	"""model = onnx.load(args.out)
	#onnx_model = OnnxModel(model)
	count = len(model.graph.initializer)
	same = [-1] * count
	for i in range(count - 1):
		if same[i] >= 0:
			continue
		for j in range(i + 1, count):
			if (model.graph.initializer[i] == model.graph.initializer[j]):
				same[j] = i
	
	for i in range(count):
		if same[i] >= 0:
			onnx_model.replace_input_of_all_nodes(model.graph.initializer[i].name,
			                                      model.graph.initializer[same[i]].name)
	
	onnx_model.update_graph()
	onnx_model.save_model_to_file(args.out)"""
	onnx.checker.check_model(onnx_model)
	
	runner.logger.info(
			'Convert successfully, save model to {}'.format(out_name))
		
	import onnxruntime
	
	ort_session = onnxruntime.InferenceSession(args.out)
	
	def to_numpy(tensor):
		return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
	
	# compute ONNX Runtime output prediction
	ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
	ort_outs = ort_session.run(None, ort_inputs)
	
	np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


if __name__ == '__main__':
	main()
