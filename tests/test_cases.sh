#!/bin/sh
python jigsaw_puzzle_solver/create_jigsaw_pieces.py testimg_1.jpg 3 3 test1 # error
python jigsaw_puzzle_solver/create_jigsaw_pieces.py sample_images/apple.png 3 3 test1 #error
python jigsaw_puzzle_solver/create_jigsaw_pieces.py -v sample_images/apple.jpg 3 3 test1 # standard
python jigsaw_puzzle_solver/create_jigsaw_pieces.py sample_images/mount_roraima.jpg 3 2 test2 --verbose # standard
python jigsaw_puzzle_solver/create_jigsaw_pieces.py sample_images/mountain_view.jpg 2 4 test3 # standard
python jigsaw_puzzle_solver/create_jigsaw_pieces.py sample_images/cybercity.jpg 1 4 test4 # edge case
python jigsaw_puzzle_solver/create_jigsaw_pieces.py sample_images/james_webb.jpg 7 7 test5 # computation heavy 1
python jigsaw_puzzle_solver/create_jigsaw_pieces.py sample_images/mountain_view.jpg 12 14 test6 # computation heavy 2 - parallel processing3

python jigsaw_puzzle_solver/solve_puzzle.py test1 -at # standard
python jigsaw_puzzle_solver/solve_puzzle.py test2 -at # standard
python jigsaw_puzzle_solver/solve_puzzle.py test3 -at # standard
python jigsaw_puzzle_solver/solve_puzzle.py test4 -at -x 1 -y 4 # edge case
python jigsaw_puzzle_solver/solve_puzzle.py test5 -at  # computation heavy 1, correct
python jigsaw_puzzle_solver/solve_puzzle.py test6 -at # computation heavy 2, parallel processing3
