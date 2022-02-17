import cv2
import numpy as np

MAX_PIECES_WINDOW_SIZE = (300, 400)  # height, width
MAX_MERGE_ANIMATION_WINDOW_SIZE = (600, 1080)  # height, width
merged_id_list = []


def start_assemble_animation(merge_history, img_cells_unaligned, img_cells, interval_millis=200):
    """
        animation routine.
    """

    if len(merge_history) < 1:
        return

    height, width = merge_history[-1]["cellblock"].size()
    for i in range(len(merge_history)):
        _display_from_cellblock(merge_history[i], img_cells_unaligned, img_cells, height, width,
                                str(i + 1) + "/" + str(len(merge_history)), True, interval_millis)
    _display_from_cellblock(merge_history[-1], img_cells_unaligned, img_cells, height, width,
                            str(len(merge_history)) + "/" + str(len(merge_history)), False, interval_millis * 4)


def _display_remaining_pieces(img_cells_unaligned, height, width, merge_id):
    """
        show remaining image fragments
    """

    global merged_id_list
    merged_id_list.append(merge_id)

    max_h = MAX_PIECES_WINDOW_SIZE[0]
    max_w = MAX_PIECES_WINDOW_SIZE[1]
    max_cell_len = max(len(img_cells_unaligned[0][0]), len(img_cells_unaligned[0][0][0]))
    mid_align_offset = (max_cell_len - min(len(img_cells_unaligned[0][0]), len(img_cells_unaligned[0][0][0]))) // 2
    window_h = int((height * max_cell_len) * 1.15)
    window_w = int((width * max_cell_len) * 1.15)
    start_h = 20
    start_w = 20
    scale = min(max_h / window_h, max_w / window_w)

    whiteboard = np.zeros((window_h, window_w, 3), dtype=np.uint8)
    whiteboard.fill(255)
    for i in range(len(img_cells_unaligned)):
        placeholder = np.full((max_cell_len, max_cell_len, 3), 245)
        paste = img_cells_unaligned[i][0]
        cell_h = len(paste)
        cell_w = len(paste[0])
        y_offset = start_h + (i // width) * int(max_cell_len * 1.1)
        x_offset = start_w + (i % width) * int(max_cell_len * 1.1)
        whiteboard[y_offset:y_offset + max_cell_len, x_offset:x_offset + max_cell_len] = placeholder
        for j in merged_id_list:
            if j == i:
                break
        else:
            if cell_h < cell_w:
                y_offset += mid_align_offset
            else:
                x_offset += mid_align_offset
            whiteboard[y_offset:y_offset + cell_h, x_offset:x_offset + cell_w] = paste
    whiteboard = cv2.resize(whiteboard, (int(window_w * scale), int(window_h * scale)))
    cv2.imshow("remaining fragments", whiteboard)


def _display_from_cellblock(merge, img_cells_unaligned, img_cells, height, width, text="", draw_borders=True,
                            interval_millis=200):
    """
        construct image from cellblock and visualize
    """

    global merged_id_list
    cellblock = merge["cellblock"].data
    celldata = merge["celldata"]
    rt, ct, rs, cs = merge["cellblock"].ht, merge["cellblock"].wt, merge["cellblock"].hs, merge["cellblock"].ws

    max_h = MAX_MERGE_ANIMATION_WINDOW_SIZE[0]
    max_w = MAX_MERGE_ANIMATION_WINDOW_SIZE[1]
    cell_h = len(img_cells[0][0])
    cell_w = len(img_cells[0][0][0])
    window_h = height * cell_h + int(cell_h * 1.0)
    window_w = width * cell_w + int(cell_w * 1.0)
    cellblock_h = (rt - rs + 1) * cell_h
    cellblock_w = (ct - cs + 1) * cell_w
    start_h = (window_h - cellblock_h) // 2
    start_w = (window_w - cellblock_w) // 2
    scale = min(max_h / window_h, max_w / window_w)

    whiteboard = np.zeros((window_h, window_w, 3), dtype=np.uint8)
    whiteboard.fill(255)

    # assemble image as specifed in cellblock blueprint
    for i in range(len(cellblock)):
        for j in range(len(cellblock[0])):
            if cellblock[i][j].is_valid():
                cdata = cellblock[i][j]
                paste = img_cells[cdata.id][cdata.transform]
                y_offset = start_h + (i - rs) * cell_h
                x_offset = start_w + (j - cs) * cell_w
                whiteboard[y_offset:y_offset + cell_h, x_offset:x_offset + cell_w] = paste
                if draw_borders and cellblock[i][j].id == celldata.id:
                    if celldata.dir == 0:
                        cv2.line(whiteboard, (x_offset, y_offset + cell_h),
                                 (x_offset + cell_w, y_offset + cell_h), (0, 0, 255), int(2 / scale))
                    elif celldata.dir == 1:
                        cv2.line(whiteboard, (x_offset, y_offset),
                                 (x_offset + cell_w, y_offset), (0, 0, 255), int(2 / scale))
                    elif celldata.dir == 2:
                        cv2.line(whiteboard, (x_offset + cell_w, y_offset),
                                 (x_offset + cell_w, y_offset + cell_h), (0, 0, 255), int(2 / scale))
                    elif celldata.dir == 3:
                        cv2.line(whiteboard, (x_offset, y_offset),
                                 (x_offset, y_offset + cell_h), (0, 0, 255), int(2 / scale))

    whiteboard = cv2.resize(whiteboard, (int(window_w * scale), int(window_h * scale)))
    whiteboard = cv2.copyMakeBorder(whiteboard, 0, 100, 0, 0, cv2.BORDER_CONSTANT)
    whiteboard = cv2.putText(whiteboard,
                             text + " similarity score =" + str(celldata.score), (30, len(whiteboard) - 50),
                             fontFace=cv2.LINE_AA, fontScale=0.5, color=(255, 255, 255))
    try:
        _display_remaining_pieces(img_cells_unaligned, height, width, celldata.id)
    except Exception as e:
        pass
    cv2.imshow("merge process", whiteboard)
    cv2.waitKey(interval_millis)
