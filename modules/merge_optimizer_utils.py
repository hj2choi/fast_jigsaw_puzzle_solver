from . import img_operations as im_op
import numpy as np
"""""
switch to parallel computation if data size reaches image count threshold
"""""
MAX_PROCESS_COUNT = 3
SWITCH_TO_PARALLEL_THRESHOLD = 128 # number of images

"""
@TODO 최적화 사용 시, 정사각형 이미지 매핑할때 특정 상황에서 이미지가 잘못된 orientation으로 붙음.
@TODO similarity matrix의 diagonality 성질 이용해서 최대 2배 최적화 가능한지 확인.

similarity matrix index 매핑 (속도 최대 4~8x 상승)

src 이미지, tgt 이미지의 border similarity 계산할 시:
일단 tgt는 고정하고 src만 이동함.
경우의 수 = rotation x flip x directions = 4 * 2 * 4 = 32
similarity_mat[src][tgt]= np.array(32)

             [0~7]
              src
  [16~23] src tgt src [24~31]
              src
             [8~15]

tgt가 고정되지 않고 transform 하면
32 x 8 = 256 경우의 수가 나옴.

아래는 tgt의 transformation 상태에 따라 similarity matrix의 index를 매핑하는 코드.
"""
MAPPING_TABLE8 = []

def generate_mapping_table8(sim_matrix):
    mapping_table = []

    for i in range(len(sim_matrix[0][1])):
        for j in range(32):
            print(i,j,"=>",sim_matrix[0][1][i],sim_matrix[0][1][j], end="")
            if (np.abs(sim_matrix[0][1][i] - sim_matrix[0][1][j]) < 0.0000001):
                print(" match")
                mapping_table.append(j)
            else:
                print()

    print(mapping_table)

def unit_test_mapping8():
    print("Transformation mapper TEST")
    flag = 1
    for i in range(32):
        if (i != t_rot(t_rot(t_rot(t_rot(i))))):
            flag = 0
            print("rotation identity test: [FAILED]")
            break
    if flag:
        print("rotation identity test: [PASSED]")

    for i in range(32):
        if (i != t_flip(t_flip(i))):
            flag = 0
            print("flip identity test: [FAILED]")
            break
    if flag:
        print("flip identity test: [PASSED]")

    for i in range(32):
        if (t_rot(t_flip(t_flip(i))) != t_flip(t_flip(t_rot(i)))):
            flag = 0
            print("commutative property test: [FAILED]")
            break
    if flag:
        print("commutative property test: [PASSED]")

'''
ex)
[0~7] => [24~31] =>
[27 24 25 26,
29 30 31 28]
'''
def t_rot(i):
    i -= 16
    if i < 0:
        if i < -8:
            i += 16
        i += 24
    return i - i%4 + (i%4 + 3 + ((i//4)%4)*2)%4

'''
ex)
[0~7] => [8~15]=>
[12 13 14 15,
8 9 10 11]
'''
def t_flip(i):
    if i < 16:
        j = i+((i//8+1)%2)*16-8
        return j - j%8 + ((j%8)+4)%8
    else:
        return i-((i%8)//4)*8+4

'''
rotation 없이 오직 flip만 사용할 시 mapping table
'''
def t_mirror4(i):
    return [1,0,3,2,5,4,7,6,13,12,15,14,9,8,11,10][i]

def t_flip4(i):
    return [6,7,4,5,2,3,0,1,10,11,8,9,14,15,12,13][i]

'''
ex)
1: rot(90, img), 2: rot(180, img), 4: rot(0, flip(img))
'''
def map8_deprecated(t_transform, s_transform):
    if t_transform >= 4:
        s_transform = t_flip(s_transform)
    for i in range(t_transform % 4):
        s_transform = t_rot(s_transform)
    return s_transform

def map8_accurate(t_transform, s_transform):
    return t_transform*8*4 + s_transform

'''
flip, mirror only
'''
def map4(t_transform, s_transform):
    return {
        0: s_transform,
        1: t_mirror4(s_transform),
        2: t_flip4(s_transform),
        3: t_mirror4(t_flip4(s_transform))
    }[t_transform]

'''
Parallel computation
'''
def init_process(img_cells_norm, t_count):
    global img_cells_norm_g, t_cnt
    img_cells_norm_g = img_cells_norm
    t_cnt = t_count

def elementwise_similarity_parallel(x):
    epoch = t_cnt * 4
    i, j, k = x[0][0], x[0][1], x[0][2]
    if (i == j):
        return 0
    return im_op.img_borders_similarity(img_cells_norm_g[j][(k%epoch)%t_cnt],
                            img_cells_norm_g[i][k//epoch], (k%epoch)//t_cnt)
