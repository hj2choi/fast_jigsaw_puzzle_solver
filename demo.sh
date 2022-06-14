python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_2.png 2 2 quickstart1 #standard
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_1.jpg 3 3 quickstart2 #standard
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_4.jpg 2 4 quickstart3 #standard
python jigsaw_puzzle_solver/fragment_image.py sample_images/testimg_3.png 7 7 quickstart4

python jigsaw_puzzle_solver/assemble_images.py quickstart1 2 2 out1
python jigsaw_puzzle_solver/assemble_images.py quickstart2 3 3 out2
python jigsaw_puzzle_solver/assemble_images.py quickstart3 2 4 out3
python jigsaw_puzzle_solver/assemble_images.py quickstart4 7 7 out7
