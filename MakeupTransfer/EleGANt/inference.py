import os
import sys

import numpy as np
import torch
from PIL import Image

# sys.path.append('.')

from .training.config import get_config
from .training.inference import Inference
from .training.utils import create_logger


class EleGANtArgs:
    def __init__(self, device, load_folder, save_folder, keepon=False):
        self.device = device
        self.load_folder = load_folder
        self.save_folder = save_folder
        self.keepon = keepon


def transfer_inference(source_dir, reference_dir, joint_mode, load_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config = get_config()
    logger = create_logger(save_path, 'EleGANt', 'info', console=True)
    logger.info(config)

    args = EleGANtArgs(device=torch.device('cuda:0'), save_folder=save_path, load_folder=load_path)
    inference = Inference(config, args, load_path)

    n_imgname = sorted(os.listdir(source_dir))
    m_imgname = sorted(os.listdir(reference_dir))

    for i, (imga_name, imgb_name) in enumerate(zip(n_imgname, m_imgname)):
        imgA = Image.open(os.path.join(source_dir, imga_name)).convert('RGB')
        imgB = Image.open(os.path.join(reference_dir, imgb_name)).convert('RGB')

        if joint_mode == 'all':
            result = inference.transfer(imgA, imgB, postprocess=True)
        elif joint_mode == 'skin':
            result = inference.joint_transfer(imgA, reference_skin=imgB, reference_lip=imgA, reference_eye=imgA,
                                              postprocess=True)
        elif joint_mode == 'lip':
            result = inference.joint_transfer(imgA, reference_skin=imgA, reference_lip=imgB, reference_eye=imgA,
                                              postprocess=True)
        elif joint_mode == 'eye':
            result = inference.joint_transfer(imgA, reference_skin=imgA, reference_lip=imgA, reference_eye=imgB,
                                              postprocess=True)
        else:
            raise Exception()

        if result is None:
            continue

        imgA = np.array(imgA)
        h, w, _ = imgA.shape
        result = result.resize((h, w))
        result = np.array(result)
        save_path2 = os.path.join(save_path, "out.png")
        Image.fromarray(result.astype(np.uint8)).save(save_path2)
