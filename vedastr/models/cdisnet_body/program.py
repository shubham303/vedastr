'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 20:32:00
Contact: yangbitao001@ke.com
'''


# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
import random
import time
import shutil
import traceback
import yaml
import logging
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

from flags import Flags


def build_config(config_path):
    #args = ArgsParser().parse_args()
    #flags = Flags(args.config).get()
    flags = Flags(config_path).get()
    log_file_path = os.path.join(flags.Global.save_model_dir, time.strftime('%Y%m%d_%H%M%S') + '.log')
    os.makedirs(flags.Global.save_model_dir, exist_ok=True)
    return flags
