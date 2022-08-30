python jigsaw_puzzle_solver/fragment_image.py sample_images/apple.jpg 3 3 demo1
python jigsaw_puzzle_solver/fragment_image.py sample_images/james_webb.jpg 2 4 demo2
python jigsaw_puzzle_solver/fragment_image.py sample_images/ocean_cam.jpg 3 5 demo3
python jigsaw_puzzle_solver/fragment_image.py sample_images/mountain_view.jpg 7 6 demo4

python jigsaw_puzzle_solver/assemble_images.py -a demo1 3 3 demo1
python jigsaw_puzzle_solver/assemble_images.py -a demo2 2 4 demo2
python jigsaw_puzzle_solver/assemble_images.py -a demo3 3 5 demo3
python jigsaw_puzzle_solver/assemble_images.py -at demo4 7 6 demo4
