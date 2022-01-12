import os
import sys
import time
from configparser import ConfigParser

from modules import assembler as asm


def main(args, cfg):
    s_time = time.time()
    images_dir = cfg.get("config", "fragments_dir")
    output_directory = cfg.get("config", "output_dir")
    enable_visualization = cfg.getboolean("config", "enable_visualization") or len(args) >= 5 and args[4] == "-v"
    animation_interval = int(cfg.get("config", "animation_interval_millis"))
    verbose = not (not cfg.getboolean("config", "debug") and not (
                len(args) >= 5 and args[4] == "-v"))  # enable verbose option for debugging
    input_filename_prefix = args[0]
    paste_cols = int(args[1])
    paste_rows = int(args[2])
    output_filename = args[3]

    # initialize
    assembler = asm.ImageAssembler.load_from_filepath(images_dir, input_filename_prefix, paste_cols, paste_rows)
    print("merge_image.py:", len(assembler.raw_imgs), "files loaded", flush=True)
    if assembler.rows * assembler.cols != len(assembler.raw_imgs):
        print("WARNING: incorrect slicing dimension.")
    if not verbose:
        sys.stdout = open(os.devnull, 'w')  # block stdout

    # main merge algorithm
    assembler.assemble()

    # save result to output directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    assembler.save_assembled_image(output_directory + "/" + output_filename)

    sys.stdout = sys.__stdout__  # restore stdout
    print("total elapsed time:", time.time() - s_time, "seconds", flush=True)
    if enable_visualization:
        assembler.start_assemble_animation(animation_interval)


if __name__ == '__main__':
    config = ConfigParser()
    config.read("config.ini")
    main(sys.argv[1:], config)
