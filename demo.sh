python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_2.png 2 2 quickstart1
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_1.jpg 3 3 quickstart2
python jigsaw_puzzle_solver/fragment_image.py sample_images/mount_roraima.jpg 3 6 quickstart3
python jigsaw_puzzle_solver/fragment_image.py sample_images/mountain_view.jpg 7 6 quickstart4

python jigsaw_puzzle_solver/assemble_images.py -a quickstart1 2 2 out1
python jigsaw_puzzle_solver/assemble_images.py -a quickstart2 3 3 out2
python jigsaw_puzzle_solver/assemble_images.py -a quickstart3 3 6 out3
python jigsaw_puzzle_solver/assemble_images.py -at quickstart4 7 6 out7
