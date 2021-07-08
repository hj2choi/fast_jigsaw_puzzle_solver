import sys, os, time
from configparser import ConfigParser
from modules import assembler as asm

def main(args, config):
    s_time = time.time()
    IMAGES_DIR = config.get("config", "fragments_dir")
    OUTPUT_DIRECTORY = config.get("config", "output_dir")
    ENABLE_VISUALIZATION = config.getboolean("config", "enable_visualization") or len(args)>=5 and args[4] == "-v"
    ANIMATION_INTERVAL = int(config.get("config", "animation_interval_millis"))
    VERBOSE = config.getboolean("config", "debug") or len(args) >=5 and args[4] == "-v" # enable VERBOSE option for debugging
    INPUT_FILENAME_PREFIX = args[0]
    PASTE_COLS = int(args[1])
    PASTE_ROWS = int(args[2])
    OUTPUT_FILENAME = args[3]

    #initialize
    assembler = asm.ImageAssembler.loadfromfilepath(IMAGES_DIR, INPUT_FILENAME_PREFIX, PASTE_COLS, PASTE_ROWS)
    print("merge_image.py:", len(assembler.raw_imgs), "files loaded", flush=True)
    if (assembler.rows * assembler.cols != len(assembler.raw_imgs)):
        print("WARNING: incorrect slicing dimension.")
    if not VERBOSE:
        sys.stdout = open(os.devnull, 'w') # block stdout

    #main merge algorithm
    assembler.assemble()

    #save result to output directory
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    assembler.save_assembled_image(OUTPUT_DIRECTORY + "/" + OUTPUT_FILENAME)

    sys.stdout = sys.__stdout__ #restore stdout
    print("total elapsed time:", time.time() - s_time, "seconds", flush=True)
    if ENABLE_VISUALIZATION:
        assembler.start_assemble_animation(ANIMATION_INTERVAL)


if __name__ == '__main__':
    config = ConfigParser()
    config.read("config.ini")
    main(sys.argv[1:], config)
