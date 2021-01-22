import sys, os, time
from configparser import ConfigParser
from modules import merger as mr

def main(args, config):
    s_time = time.time()
    IMAGES_DIR = config.get("config", "images_dir")
    OUTPUT_DIRECTORY = config.get("config", "output_dir")
    ENABLE_VISUALIZATION = config.getboolean("config", "enable_merge_visualization") or len(args)>=5 and args[4] == "-v"
    ANIMATION_INTERVAL = int(config.get("config", "animation_interval_millis"))
    VERBOSE = config.getboolean("config", "debug") or len(args)>=5 and args[4] == "-v" # enable VERBOSE option for debugging
    INPUT_FILENAME_PREFIX = args[0]
    PASTE_COLS = int(args[1])
    PASTE_ROWS = int(args[2])
    OUTPUT_FILENAME = args[3]

    #initialize
    merger = mr.ImageMerger.loadfromfilepath(IMAGES_DIR, INPUT_FILENAME_PREFIX, PASTE_COLS, PASTE_ROWS)
    print("merge_image.py:",len(merger.img_cells),"files loaded")
    if (merger.rows*merger.cols != len(merger.img_cells)):
        print("WARNING: incorrect slicing dimension.")

    if not VERBOSE:
        sys.stdout = open(os.devnull, 'w') # block stdout

    #main merge algorithm
    merger.merge()

    #save result to output directory
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    merger.save_merged_image(OUTPUT_DIRECTORY+"/"+OUTPUT_FILENAME)

    sys.stdout = sys.__stdout__ #restore stdout
    print("total elapsed time:",time.time()-s_time,"seconds")
    if ENABLE_VISUALIZATION:
        merger.start_merge_process_animation(ANIMATION_INTERVAL)


if __name__ == '__main__':
    config = ConfigParser()
    config.read("config.ini")
    main(sys.argv[1:], config)
