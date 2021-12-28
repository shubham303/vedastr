import argparse
import os
import sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import cv2  # noqa 402
from vedastr.runners import InferenceRunner  # noqa 402
from vedastr.utils import Config  # noqa 402


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

    aug = runner.transform(image=image, label='')
    image, label = aug['image'], aug['label']  # noqa 841
    image = image.unsqueeze(0).cuda()
    dummy_input = (image, runner.converter.test_encode([''])[0])
    model = runner.model.cuda().eval()
    need_text = runner.need_text
    if not need_text:
        dummy_input = dummy_input[0]
    
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    
    
    import onnx

    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)
   
    runner.logger.info(
        'Convert successfully, save model to {}'.format(out_name))


if __name__ == '__main__':
    main()