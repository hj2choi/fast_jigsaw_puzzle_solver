#!/bin/sh
python cut_image.py testimg_1.jpg 3 3 test1 # error
python cut_image.py test_images/testimg_1.png 3 3 test1 #error

python cut_image.py test_images/testimg_2 2 2 test1 #standard
python cut_image.py test_images/testimg_1.jpg 3 3 test2 #standard
python cut_image.py test_images/testimg_3 2 4 test3 #standard
python cut_image.py test_images/testimg_1 4 1 test4 # edge case
python cut_image.py test_images/testimg_5 2 2 test5 # square images
python cut_image.py test_images/testimg_4 12 14 test6 # computation heavy 1
python cut_image.py test_images/testimg_5 7 7 test7 # computation heavy 2

python merge_image.py test1 2 2 out1
python merge_image.py test2 3 3 out2
python merge_image.py test3 2 4 out3
python merge_image.py test4 4 1 out4
python merge_image.py test5 2 2 out5
python merge_image.py test6 12 14 out6
python merge_image.py test7 7 7 out7
python merge_image.py test3 4 5 out8 # error case 1
python merge_image.py test3 2 2 out9 # error case 2
