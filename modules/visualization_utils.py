import cv2
import numpy as np
from . import img_operations as im_op


MAX_PIECES_WINDOW_SIZE = (280,400) # height, width
#MAX_MERGE_ANIMATION_WINDOW_SIZE = (768, 1024) # height, width
MAX_MERGE_ANIMATION_WINDOW_SIZE = (450, 750) # height, width

"""
visualize image pair stitching at given orientation and stitching direction.
"""
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
"""
find actual cellblock width and height
"""
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

"""
animation routine.
"""
def start_merge_process_animation(merge_history, img_cells, rows, cols, t_cnt, windowlock = False, interval_millis = 500):
    if (len(merge_history) < 1):
        return

    rows, cols = resolve_row_col(merge_history[-1]["next_cblock"])
    init_cellblock = np.copy(merge_history[0]["next_cblock"])
    init_cellblock.fill(-1)
    init_cellblock[rows][cols].fill(int(0))
    display_from_cellblock({"next_cblock":init_cellblock, "score":1.0, "id":0, "dir":0}, img_cells, rows, cols, t_cnt, windowlock,interval_millis)
    for i in range(len(merge_history)-1):
        display_from_cellblock(merge_history[i], img_cells, rows, cols, t_cnt, windowlock,interval_millis)
    merge_history[-1]["dir"] = -1
    display_from_cellblock(merge_history[-1], img_cells, rows, cols, t_cnt, True,interval_millis)

"""


"""
MERGED_ID_LIST = []
def display_remaining_pieces(cellblock, img_cells, rows, cols, merge_id, windowlock = False, interval_millis = 500):
    global MERGED_ID_LIST
    MERGED_ID_LIST.append(merge_id)

    max_h = MAX_PIECES_WINDOW_SIZE[0]
    max_w = MAX_PIECES_WINDOW_SIZE[1]
    cell_h = len(img_cells[0][0])
    cell_w = len(img_cells[0][0][0])
    window_h = int((rows * cell_h)*1.15)
    window_w = int((cols * cell_w)*1.15)
    start_h = 20
    start_w = 20
    scale = min(max_h/window_h, max_w/window_w)

    whiteboard = np.zeros((window_h, window_w, 3), dtype = np.uint8)
    whiteboard.fill(255)

    for i in range(len(img_cells)):
        for j in MERGED_ID_LIST:
            if j == i:
                break
        else:
            paste = img_cells[i][0]
            y_offset = start_h+(i//cols)*int(cell_h*1.1)#+int(cell_h/10)
            x_offset = start_w+(i%cols)*int(cell_w*1.1)#+int(cell_w/10)
            whiteboard[y_offset:y_offset+cell_h, x_offset:x_offset+cell_w] = paste

    whiteboard = cv2.resize(whiteboard, (int(window_w*scale), int(window_h*scale)))
    cv2.imshow("remaining fragments", whiteboard)
    #if windowlock:
    #    cv2.waitKey(interval_millis*5)
    #else:
    #    cv2.waitKey(interval_millis)



"""
construct image from cellblock and visualize
"""
def display_from_cellblock(merge, img_cells, rows, cols, t_cnt, windowlock = False, interval_millis = 500):
    global MERGED_ID_LIST
    cellblock = merge["next_cblock"]
    merge_id = merge["id"]
    merge_dir = merge["dir"]
    rt, ct, rs, cs = 0, 0, len(cellblock), len(cellblock[0])
    for i in range(len(cellblock)):
        for j in range(len(cellblock[0])):
            if cellblock[i][j][0] > -1:
                if i < rs: rs = i
                if i > rt: rt = i
                if j < cs: cs = j
                if j > ct: ct = j

    max_h = MAX_MERGE_ANIMATION_WINDOW_SIZE[0]
    max_w = MAX_MERGE_ANIMATION_WINDOW_SIZE[1]
    cell_h = len(img_cells[0][0])
    cell_w = len(img_cells[0][0][0])
    window_h = rows * cell_h + cell_h*1
    window_w = cols * cell_w + cell_w*1
    cellblock_h = (rt - rs + 1)*cell_h
    cellblock_w = (ct - cs + 1)*cell_w
    start_h = (window_h - cellblock_h) // 2
    start_w = (window_w - cellblock_w) // 2
    scale = min(max_h/window_h, max_w/window_w)

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
                if cellblock[i][j][0] == merge_id:
                    if merge_dir == 0:
                        cv2.line(whiteboard,(x_offset,y_offset+cell_h),(x_offset+cell_w,y_offset+cell_h),(0,0,255),int(2/scale))
                    elif merge_dir == 1:
                        cv2.line(whiteboard,(x_offset,y_offset),(x_offset+cell_w,y_offset),(0,0,255),int(2/scale))
                    elif merge_dir == 2:
                        cv2.line(whiteboard,(x_offset+cell_w,y_offset),(x_offset+cell_w,y_offset+cell_h),(0,0,255),int(2/scale))
                    elif merge_dir == 3:
                        cv2.line(whiteboard,(x_offset,y_offset),(x_offset,y_offset+cell_h),(0,0,255),int(2/scale))

    whiteboard = cv2.resize(whiteboard, (int(window_w*scale), int(window_h*scale)))
    whiteboard = cv2.copyMakeBorder( whiteboard, 0, 100, 0, 0, cv2.BORDER_CONSTANT)
    whiteboard = cv2.putText(whiteboard, "similarity score ="+str(merge["score"]), (30, len(whiteboard)-50), fontFace = cv2.LINE_AA, fontScale = 0.5, color = (255,255,255))
    try:
        display_remaining_pieces(cellblock, img_cells, rows, cols, merge["id"])
    except:
        pass
    cv2.imshow("merge process", whiteboard)
    if windowlock:
        cv2.waitKey(interval_millis*5)
    else:
        cv2.waitKey(interval_millis)
