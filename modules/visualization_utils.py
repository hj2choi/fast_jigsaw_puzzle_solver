import cv2
import numpy as np
from . import img_operations as im_op

def img_borders_similarity(img1, img2, dir, waitkey = 100, text = ""):
    sim = img_borders_similarity(img1, img2, dir)

    if (visualize):
        disp = {
            DIR['d']: np.concatenate((img1, img2)),
            DIR['u']: np.concatenate((img2, img1)),
            DIR['r']: np.concatenate((img1, img2), axis = 1),
            DIR['l']: np.concatenate((img2, img1), axis = 1)
        }[dir]

        #disp = np.concatenate((disp, np.zeros((300, len(disp[0]), 3))))
        disp = cv2.copyMakeBorder( disp, 0, 3, 0, 0, cv2.BORDER_CONSTANT)
        cv2.line(disp,(0,len(disp)),(len(disp[0]),len(disp)),(255,0,0),3)
        disp = cv2.copyMakeBorder( disp, 0, 100, 0, 0, cv2.BORDER_CONSTANT)

        disp = cv2.putText(disp, text+" "+str(sim), (30, len(disp)-50), fontFace = cv2.LINE_AA, fontScale = 0.5, color = (255,255,255))

        if dir < 2:
            cv2.line(disp,(0,len(img1)),(len(img1[0]),len(img1)),(255,0,0),1)
        else:
            cv2.line(disp,(len(img1[0]),0),(len(img1[0]),len(img1)),(255,0,0),1)
        cv2.imshow('image',disp)
        if windowlock:
            cv2.waitKey(0)
        else:
            cv2.waitKey(50)


def resolve_row_col(cellblock):
    rt, ct, rs, cs = 0, 0, len(cellblock), len(cellblock[0])
    for i in range(len(cellblock)):
        for j in range(len(cellblock[0])):
            if cellblock[i][j][0] > -1:
                if i < rs: rs = i
                if i > rt: rt = i
                if j < cs: cs = j
                if j > ct: ct = j
    return (rt - rs + 1), (ct - cs + 1)


def start_merge_process_animation(merge_history, img_cells, rows, cols, t_cnt, windowlock = False, interval_millis = 500):
    if (len(merge_history) < 1):
        return

    rows, cols = resolve_row_col(merge_history[-1]["next_cblock"])
    init_cellblock = np.copy(merge_history[0]["next_cblock"])
    init_cellblock.fill(-1)
    init_cellblock[rows][cols].fill(int(0))
    display_from_cellblock({"next_cblock":init_cellblock, "score":1.0}, img_cells, rows, cols, t_cnt, windowlock,interval_millis)
    for i in range(len(merge_history)-1):
        display_from_cellblock(merge_history[i], img_cells, rows, cols, t_cnt, windowlock,interval_millis)
    display_from_cellblock(merge_history[-1], img_cells, rows, cols, t_cnt, True,interval_millis)


def display_from_cellblock(merge, img_cells, rows, cols, t_cnt, windowlock = False, interval_millis = 500):
    cellblock = merge["next_cblock"]
    rt, ct, rs, cs = 0, 0, len(cellblock), len(cellblock[0])
    for i in range(len(cellblock)):
        for j in range(len(cellblock[0])):
            if cellblock[i][j][0] > -1:
                if i < rs: rs = i
                if i > rt: rt = i
                if j < cs: cs = j
                if j > ct: ct = j

    cell_h = len(img_cells[0][0])
    cell_w = len(img_cells[0][0][0])
    window_h = rows * cell_h + cell_h*2
    window_w = cols * cell_w + cell_w*2
    cellblock_h = (rt - rs + 1)*cell_h
    cellblock_w = (ct - cs + 1)*cell_w
    start_h = (window_h - cellblock_h) // 2
    start_w = (window_w - cellblock_w) // 2

    whiteboard = np.zeros((window_h, window_w, 3), dtype = np.uint8)
    whiteboard.fill(255)

    # assemble image as specifed in cellblock blueprint
    for i in range(len(cellblock)):
        for j in range(len(cellblock[0])):
            if (cellblock[i][j][0] > -1):
                idx = cellblock[i][j]
                paste = img_cells[idx[0]][idx[1] % t_cnt]
                y_offset = start_h+(i-rs)*cell_h
                x_offset = start_w+(j-cs)*cell_w
                whiteboard[y_offset:y_offset+cell_h, x_offset:x_offset+cell_w] = paste


    max_h = 768
    max_w = 1024
    scale = min(max_h/window_h, max_w/window_w)
    whiteboard = cv2.resize(whiteboard, (int(window_w*scale), int(window_h*scale)))
    whiteboard = cv2.copyMakeBorder( whiteboard, 0, 100, 0, 0, cv2.BORDER_CONSTANT)
    whiteboard = cv2.putText(whiteboard, "similarity score ="+str(merge["score"]), (30, len(whiteboard)-50), fontFace = cv2.LINE_AA, fontScale = 0.5, color = (255,255,255))

    cv2.imshow("current cellblock", whiteboard)
    if windowlock:
        cv2.waitKey(interval_millis*5)
    else:
        cv2.waitKey(interval_millis)
