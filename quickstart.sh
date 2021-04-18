python fragment_image.py test_images/testimg_2 2 2 test1 #standard
python fragment_image.py test_images/testimg_1.jpg 3 3 test2 #standard
python fragment_image.py test_images/testimg_4 2 4 test3 #standard
python fragment_image.py test_images/testimg_3 7 7 test7

python merge_image.py test1 2 2 out1 -v
python merge_image.py test2 3 3 out2 -v
python merge_image.py test3 2 4 out3 -v
python merge_image.py test7 7 7 out7 -v
