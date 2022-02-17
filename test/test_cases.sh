#!/bin/sh
python src/fragment_image.py testimg_1.jpg 3 3 test1 # error
python src/fragment_image.py test_images/testimg_1.png 3 3 test1 #error

python src/fragment_image.py test_images/testimg_2 2 2 test1 # standard
python src/fragment_image.py test_images/testimg_1.jpg 3 3 test2 # standard
python src/fragment_image.py test_images/testimg_4 2 4 test3 # standard
python src/fragment_image.py test_images/testimg_1 4 1 test4 # edge case
python src/fragment_image.py test_images/testimg_5 2 2 test5 # square images
python src/fragment_image.py test_images/testimg_3 7 7 test6 # computation heavy 1
python src/fragment_image.py test_images/testimg_3 10 10 test7 # computation heavy 2
python src/fragment_image.py test_images/testimg_3 12 14 test8 # computation heavy 3

python src/assemble_fragments.py test1 2 2 out1 # standard
python src/assemble_fragments.py test2 3 3 out2 # standard
python src/assemble_fragments.py test3 2 4 out3 # standard
python src/assemble_fragments.py test4 4 1 out4 # edge case
python src/assemble_fragments.py test5 2 2 out5 # black-white square
python src/assemble_fragments.py test6 7 7 out6  # computation heavy 1, correct
python src/assemble_fragments.py test7 10 10 out7 # computation heavy 2, correct
python src/assemble_fragments.py test8 12 14 out8 # computation heavy 3, incorrect
python src/assemble_fragments.py test3 4 5 out9 # error case 1
python src/assemble_fragments.py test3 2 2 out10 # error case 2
