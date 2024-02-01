import os
from os.path import join, basename, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import GPDM
from patch_swd import PatchSWDLoss
from utils import read_data, dump_images, get_pyramid_scales, show_nns
import argparse


mvtectAD = "/raid/zhangss/dataset/ADetection/mvtecAD"
mvtectTexture = ["grid"] # "carpet", "grid"
mvtectTexture1 = ["tile", "wood"]
mvtectObject1 = ["bottle", "cable", "capsule", ]

mvtectObject2 = ["pill", "screw", "toothbrush"]
mvtectObject3 = ["hazelnut", "metal_nut", "transistor", "zipper"]

def parse_args():
    # IO
    parser = argparse.ArgumentParser(description='Run GPDM')
    parser.add_argument('--target_image', help="Image or a directory with images for reference patch distribution to be matched", 
                        default="/raid/zhangss/dataset/ADetection/mvtecAD/carpet/test/color/")
    parser.add_argument('--max_inputs', default=None, type=int, help="if target is a directory, this limits the number of images used")
    parser.add_argument('--output_dir', default="outputs", help="Where to put the results")
    parser.add_argument('--debug', action='store_true', default=False, help="Dump debug images")
    parser.add_argument('--device', default="cuda:4")
    parser.add_argument('--gray_scale', default=False, action='store_true', help="Convert inputs to gray scale")

    # SWD parameters
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--num_proj', type=int, default=64, help="Number of random projections used to approximate SWD")

    # Pyramids parameters
    parser.add_argument('--fine_dim', type=int, default=None,
                        help="Height of the largest ptramid scale (can be used to get smaller output)."
                             "If None use the target_image height")
    parser.add_argument('--coarse_dim', type=int, default=21,
                        help="Height of the smallest pyramid scale, When starting from noise,"
                             " bigger coarse dim lets the images outputs go more diverse (coarse_dim==~patch_size) "
                             "will probably output a copy the input")
    parser.add_argument('--pyr_factor', type=float, default=0.75, help="Downscale factor of the pyramid")
    parser.add_argument('--height_factor', type=float, default=1.,
                        help="Controls the aspect ratio of the result: factor of height")
    parser.add_argument('--width_factor', type=float, default=1.,
                        help="Controls the aspect ratio of the result: factor of width")

    # GPDM parameters
    parser.add_argument('--init_from', default='zeros',
                        help="Defines the intial guess for the first level. Can one of ('zeros', 'target', '<path-to-image>')")
    parser.add_argument('--lr', type=float, default=0.01, help="Adam learning rate for the optimization")
    parser.add_argument('--num_steps', type=int, default=300, help="Number of Adam steps")
    parser.add_argument('--noise_sigma', type=float, default=1.5, help="Std of noise added to the first initial image")
    parser.add_argument('--num_outputs', type=int, default=72,
                        help="If > 1, batched inference is used (see paper) and multiple images are generated")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.max_inputs = 1

    for target in mvtectTexture:
        
        # 获取文件夹下的所有文件名
        target_path = join(mvtectAD, target, "test")
        file_names  = os.listdir(target_path)
        for file_name in file_names:

            if file_name != "broken":
                continue

            args.target_image = join(target_path, file_name)
            print(args.target_image)

            refernce_images = read_data(args.target_image, args.max_inputs, args.gray_scale)
            criteria        = PatchSWDLoss(patch_size=args.patch_size, stride=args.stride, num_proj=args.num_proj, c=refernce_images.shape[1])
            fine_dim        = args.fine_dim if args.fine_dim is not None else refernce_images.shape[-2]
            outputs_dir     = join(args.output_dir, target, file_name)
            os.makedirs(outputs_dir, exist_ok=True)

            iter_nums       = int(500 / args.num_outputs) + 1
            for i in range(iter_nums):
                new_images, last_lvl_references = GPDM.generate(refernce_images, criteria,
                                                                pyramid_scales=get_pyramid_scales(fine_dim, args.coarse_dim, args.pyr_factor),
                                                                aspect_ratio=(args.height_factor, args.width_factor),
                                                                init_from=args.init_from,
                                                                lr=args.lr,
                                                                num_steps=args.num_steps,
                                                                additive_noise_sigma=args.noise_sigma,
                                                                num_outputs=args.num_outputs,
                                                                debug_dir=f"{outputs_dir}/debug" if args.debug else None,
                                                                device=args.device)
            
                for j in range(len(new_images)):
                    out_file_name = f"output_{i*args.num_outputs+j+1:06d}.png"
                    dump_images(new_images[j], outputs_dir, out_file_name)