import argparse
import datetime
import os
import shutil
import subprocess
import sys

from modules.paths import models_path
from .EleGANt import inference as elegant
from .face_parsing import test as face_parsing

python = sys.executable


def inference(work_dir, method, target_image, template_image, size=288, template_image_seg=None, joint='all'):
    # Prepare work folders
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_dir = os.path.join(work_dir, current_time)
    img_dir = os.path.join(root_dir, 'images')
    img_makeup_dir = os.path.join(img_dir, 'makeup')
    img_nonmakeup_dir = os.path.join(img_dir, 'non-makeup')
    seg_dir = os.path.join(root_dir, 'seg1')
    seg_makeup_dir = os.path.join(seg_dir, 'makeup')
    seg_nonmakeup_dir = os.path.join(seg_dir, 'non-makeup')
    output_dir = os.path.join(root_dir, 'result')
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(img_makeup_dir, exist_ok=True)
    os.makedirs(img_nonmakeup_dir, exist_ok=True)
    os.makedirs(seg_makeup_dir, exist_ok=True)
    os.makedirs(seg_nonmakeup_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Copy images
    shutil.copy(target_image, img_nonmakeup_dir)
    shutil.copy(template_image, img_makeup_dir)
    template_image_seg_valid = False
    if template_image_seg and os.path.isfile(template_image_seg):
        template_image_seg_valid = True
        shutil.copy(template_image_seg, seg_makeup_dir)

    # Generate face annotation
    if method == 'SSAT':
        face_parsing.evaluate(respth=seg_nonmakeup_dir, dspth=img_nonmakeup_dir, cp='79999_iter.pth')
        if not template_image_seg_valid:
            face_parsing.evaluate(respth=seg_makeup_dir, dspth=img_makeup_dir, cp='79999_iter.pth')

    # Transfer
    if method == 'SSAT':
        run_cmd = [python, f'{os.path.join("extensions", "adetailer", "MakeupTransfer", "SSAT", "test.py")}']
        run_cmd.append(f'--dataroot={img_dir}')
        run_cmd.append(f'--checkpoint_dir={os.path.join(models_path, "adetailer")}')
        run_cmd.append(f'--result_dir={output_dir}')
        run_cmd.append(f'--resize_size={size}')
        run_cmd.append(f'--crop_size={size}')
    elif method == 'PSGAN':
        run_cmd = ""  # not used, replaced by EleGANt
    elif method == 'EleGANt':
        elegant.transfer_inference(source_dir=img_nonmakeup_dir, reference_dir=img_makeup_dir, joint_mode=joint,
                                   load_path=os.path.join(models_path, "adetailer", "sow_pyramid_a5_e3d2_remapped.pth"),
                                   save_path=output_dir)
        return output_dir
    else:
        run_cmd = ""
    p = subprocess.run(run_cmd, shell=True)
    if p.returncode == 0:
        return output_dir
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--work_dir',
        dest='work_dir',
        type=str,
        required=True)
    parser.add_argument(
        '--target_image',
        dest='target_image',
        type=str,
        required=True)
    parser.add_argument(
        '--template_image',
        dest='template_image',
        type=str,
        required=True)
    parser.add_argument(
        '--template_image_seg',
        dest='template_image_seg',
        type=str,
        default=None)
    parser.add_argument(
        '--size',
        dest='size',
        type=int,
        default=288)

    args = parser.parse_args()
    inference(args.work_dir, args.target_image, args.template_image, args.size,
              template_image_seg=args.template_image_seg)
