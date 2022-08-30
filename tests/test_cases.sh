#!/bin/sh
python jigsaw_puzzle_solver/fragment_image.py testimg_1.jpg 3 3 test1 # error
python jigsaw_puzzle_solver/fragment_image.py sample_images/apple.png 3 3 test1 #error
python jigsaw_puzzle_solver/fragment_image.py -v sample_images/apple.jpg 3 3 test1 # standard
python jigsaw_puzzle_solver/fragment_image.py sample_images/mount_roraima.jpg 3 2 test2 --verbose # standard
python jigsaw_puzzle_solver/fragment_image.py sample_images/mountain_view.jpg 2 4 test3 # standard
python jigsaw_puzzle_solver/fragment_image.py sample_images/cybercity.jpg 1 6 test4 # edge case
python jigsaw_puzzle_solver/fragment_image.py sample_images/james_webb.jpg 7 7 test5 # computation heavy 1
python jigsaw_puzzle_solver/fragment_image.py sample_images/mountain_view.jpg 12 14 test6 # computation heavy 2 - parallel processing3

python jigsaw_puzzle_solver/assemble_images.py test1 3 3 test1 # standard
python jigsaw_puzzle_solver/assemble_images.py test2 2 3 test2 # standard
python jigsaw_puzzle_solver/assemble_images.py test3 2 4 test3 # standard
python jigsaw_puzzle_solver/assemble_images.py test4 1 6 test4 # edge case
python jigsaw_puzzle_solver/assemble_images.py test5 7 7 test5  # computation heavy 1, correct
python jigsaw_puzzle_solver/assemble_images.py test6 12 14 test6 # computation heavy 2, parallel processing3
