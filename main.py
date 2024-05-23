import os
from os.path import join, basename, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import GPDM
from patch_swd import PatchSWDLoss
from utils import read_data, dump_images, get_pyramid_scales, show_nns
import argparse

def parse_args():
    # IO
    parser = argparse.ArgumentParser(description='Run GPDM')
    
    # 需要配置 原始图片文件夹路径
    parser.add_argument('--target_path', default="datasets", help="directory with images for reference patch distribution to be matched")
    
    # 需要配置 使用多少张图像进行生成
    parser.add_argument('--max_inputs',  default=1, type=int, help="if target is a directory, this limits the number of images used")
    
    # 需要配置 输出图片的路径
    parser.add_argument('--output_dir', default="outputs", help="Where to put the results")
    
    # 需要配置 使用哪个GPU
    parser.add_argument('--device', default="cuda:0")
    
    # 需要配置 是否是灰度图/彩色图
    parser.add_argument('--gray_scale', default=True, help="Convert inputs to gray scale")

    # SWD parameters
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--stride',     type=int, default=1)
    parser.add_argument('--num_proj',   type=int, default=64, help="Number of random projections used to approximate SWD")

    # 高级参数 先不管 Pyramids parameters
    parser.add_argument('--debug', action='store_true', default=False, help="Dump debug images")
    parser.add_argument('--fine_dim', type=int, default=None, help="Height of the largest ptramid scale (can be used to get smaller output)."
                             "If None use the target_image height")
    parser.add_argument('--coarse_dim', type=int, default=31,
                        help="Height of the smallest pyramid scale, When starting from noise,"
                             " bigger coarse dim lets the images outputs go more diverse (coarse_dim==~patch_size) "
                             "will probably output a copy the input")
    parser.add_argument('--pyr_factor',    type=float, default=0.75, help="Downscale factor of the pyramid")
    parser.add_argument('--height_factor', type=float, default=1.,   help="Controls the aspect ratio of the result: factor of height")
    parser.add_argument('--width_factor',  type=float, default=1.,   help="Controls the aspect ratio of the result: factor of width")

    # GPDM parameters
    # 需要配置 zeros         --- 从0生成图像，不确定性强
    #         target        --- 从输入的图像生成，确定性强
    #         path-to-image --- 结合path-to-image的轮廓和target的细节信息生成
    parser.add_argument('--init_from',   default='target', help="Defines the intial guess for the first level. Can one of ('zeros', 'target', '<path-to-image>')")
    parser.add_argument('--lr',          type=float, default=0.01, help="Adam learning rate for the optimization")
    parser.add_argument('--num_steps',   type=int, default=300, help="Number of Adam steps")
    
    # 需要配置 值越大，生成的图像的差异越大
    parser.add_argument('--noise_sigma', type=float, default=1.5, help="Std of noise added to the first initial image")
    
    # 一次生成多少张图像
    parser.add_argument('--num_outputs', type=int, default=2,  help="If > 1, batched inference is used (see paper) and multiple images are generated")
    
    # 一共生成多少张
    parser.add_argument('--num_total',   type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 获取文件夹下的所有文件名
    file_names  = os.listdir(args.target_path)
    for file_name in file_names:

        args.target_image = join(args.target_path, file_name)
        print(args.target_image)

        refernce_images = read_data(args.target_image, args.max_inputs, args.gray_scale)
        criteria        = PatchSWDLoss(patch_size=args.patch_size, stride=args.stride, num_proj=args.num_proj, c=refernce_images.shape[1])
        fine_dim        = args.fine_dim if args.fine_dim is not None else refernce_images.shape[-2]
        file_base_name  = basename(args.target_image).split('.')[0]
        outputs_dir     = join(args.output_dir, "industry", file_base_name)
        os.makedirs(outputs_dir, exist_ok=True)

        iter_nums       = int(args.num_total / args.num_outputs) + 1
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
                os.makedirs(outputs_dir, exist_ok=True)
                out_file_name = f"output_{i*args.num_outputs+j+1:06d}.png"
                dump_images(new_images[j], outputs_dir, out_file_name)