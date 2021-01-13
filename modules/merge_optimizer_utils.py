from . import img_operations as im_op
import numpy as np
"""""
switch to parallel computation if data size reaches image count threshold
"""""
MAX_PROCESS_COUNT = 3
SWITCH_TO_PARALLEL_THRESHOLD = 128 # number of images

"""
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
MAPPING_TABLE8 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                27, 24, 25, 26, 31, 28, 29, 30, 19, 16, 17, 18, 23, 20, 21, 22, 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14,
                10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 26, 27, 24, 25, 30, 31, 28, 29, 18, 19, 16, 17, 22, 23, 20, 21,
                17, 18, 19, 16, 21, 22, 23, 20, 25, 26, 27, 24, 29, 30, 31, 28, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4,
                12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1, 20, 23, 22, 21, 16, 19, 18, 17, 28, 31, 30, 29, 24, 27, 26, 25,
                29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
                6, 5, 4, 7, 2, 1, 0, 3, 14, 13, 12, 15, 10, 9, 8, 11, 30, 29, 28, 31, 26, 25, 24, 27, 22, 21, 20, 23, 18, 17, 16, 19,
                23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]

def generate_mapping_table8(sim_matrix):
    mapping_table = []
    for i in range(len(sim_matrix[0][1])):
        flag = 0
        for j in range(32):
            print(i,j,"=>",sim_matrix[0][2][i],sim_matrix[0][2][j], end="")
            if (np.abs(sim_matrix[0][2][i] - sim_matrix[0][2][j]) < 0.0000001):
                if not flag:
                    print(" match", end="")
                    flag = 1
                    mapping_table.append(j)
                else:
                    print("\n SOMETHING's WRONG")
                    return
            print()
        print()

    print(mapping_table)

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
def map8(t_transform, s_transform):
    return MAPPING_TABLE8[t_transform*8*4 + s_transform]

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
