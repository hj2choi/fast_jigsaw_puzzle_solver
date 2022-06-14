#!/bin/sh
python jigsaw_puzzle_solver/fragment_image.py testimg_1.jpg 3 3 test1 # error
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_1.png 3 3 test1 #error

python jigsaw_puzzle_solver/fragment_image.py -v sample_images/testimg_2.png 2 2 test1 # standard
python jigsaw_puzzle_solver/fragment_image.py --verbose sample_images/testimg_1.jpg 3 3 test2 # standard
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_4.jpg 2 4 test3 # standard
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_1.jpg 4 1 test4 -v # edge case
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_5.png -v 2 2 test5 # square images
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_3.png 7 7 test6 --verbose # computation heavy 1
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_3.png 10 10 test7 # computation heavy 2
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_3.png --verbose 12 14 test8 # computation heavy 3

python jigsaw_puzzle_solver/assemble_images.py test1 2 2 out1 # standard
python jigsaw_puzzle_solver/assemble_images.py test2 3 3 out2 # standard
python jigsaw_puzzle_solver/assemble_images.py test3 2 4 out3 # standard
python jigsaw_puzzle_solver/assemble_images.py test4 4 1 out4 # edge case
python jigsaw_puzzle_solver/assemble_images.py test5 2 2 out5 # black-white square
python jigsaw_puzzle_solver/assemble_images.py test6 7 7 out6  # computation heavy 1, correct
python jigsaw_puzzle_solver/assemble_images.py test7 10 10 out7 # computation heavy 2, correct
python jigsaw_puzzle_solver/assemble_images.py test8 12 14 out8 # computation heavy 3, incorrect
python jigsaw_puzzle_solver/assemble_images.py test3 4 5 out9 # error case 1
python jigsaw_puzzle_solver/assemble_images.py test3 2 2 out10 # error case 2
