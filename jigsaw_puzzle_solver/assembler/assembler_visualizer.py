""" assembler_visualizer.py
assembler.py helper routines for visualizing images assembly animation.

note: cv2.imshow() works best when kept in the main thread.
"""

import cv2
import numpy as np

MAX_PIECES_WINDOW_SIZE = (300, 400)  # height, width
MAX_MERGE_ANIMATION_WINDOW_SIZE = (600, 1080)  # height, width


def start_assembly_animation(merge_history, img_cells_unaligned, img_cells,
                             show_spanning_tree=False, interval_millis=200):
    """
    animation routine.

    Args:
        merge_history: list of dict, [{cellblock, celldata}]
        img_cells_unaligned: list of cv2 images
        img_cells: list of cv2 images
        show_spanning_tree: bool
        interval_millis: int
    """
    merged_id_list = []

    if not merge_history:
        return
    rows, cols = merge_history[-1]["cellblock"].block_size()

    if show_spanning_tree:
        for i, merge in enumerate(merge_history):
            # longer interval for last iteration
            interval_multiplier = 1 if i < len(merge_history) - 1 else 4
            _display_remaining_pieces(img_cells_unaligned, merged_id_list, rows, cols,
                                      merge["celldata"].img_id)
            _display_minimum_spanning_tree(merge, img_cells, rows, cols,
                                           str(i + 1) + "/" + str(len(merge_history)),
                                           interval_multiplier * interval_millis)

    else:
        for i, merge in enumerate(merge_history):
            # longer interval for last iteration
            interval_multiplier = 1 if i < len(merge_history) - 1 else 4
            _display_remaining_pieces(img_cells_unaligned, merged_id_list, rows, cols,
                                      merge["celldata"].img_id)
            _display_current_assembly_progress(merge, img_cells, rows, cols,
                                               str(i + 1) + "/" + str(len(merge_history)),
                                               i < len(merge_history) - 1,
                                               interval_multiplier * interval_millis)

    cv2.destroyAllWindows()


def _display_remaining_pieces(img_cells_unaligned, merged_id_list, rows, cols, merge_id):
    """
    Private method, shows remaining image fragments.
    """
    merged_id_list.append(merge_id)

    max_h = MAX_PIECES_WINDOW_SIZE[0]
    max_w = MAX_PIECES_WINDOW_SIZE[1]
    max_cell_len = max(len(img_cells_unaligned[0][0]), len(img_cells_unaligned[0][0][0]))
    mid_align_offset = (max_cell_len -
                        min(len(img_cells_unaligned[0][0]), len(img_cells_unaligned[0][0][0]))) // 2
    window_h = int((rows * max_cell_len) * 1.15)
    window_w = int((cols * max_cell_len) * 1.15)
    start_h = 20
    start_w = 20
    scale = min(max_h / window_h, max_w / window_w)

    whiteboard = np.zeros((window_h, window_w, 3), dtype=np.uint8)
    whiteboard.fill(255)
    for i, _ in enumerate(img_cells_unaligned):
        placeholder = np.full((max_cell_len, max_cell_len, 3), 245)
        paste = img_cells_unaligned[i][0]
        cell_h = len(paste)
        cell_w = len(paste[0])
        y_offset = start_h + (i // cols) * int(max_cell_len * 1.1)
        x_offset = start_w + (i % cols) * int(max_cell_len * 1.1)

        try:
            whiteboard[y_offset:y_offset + max_cell_len,
                        x_offset:x_offset + max_cell_len] = placeholder
        except ValueError:
            continue
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
    cv2.imshow("image fragments", whiteboard)


def _display_current_assembly_progress(merge, img_cells, rows, cols, text="", draw_borders=True,
                                       interval_millis=200):
    """
    Private method, shows remaining image fragments.
    """
    cellblock = merge["cellblock"].data
    celldata = merge["celldata"]
    bottom, right, top, left = merge["cellblock"].top, merge["cellblock"].right, \
                               merge["cellblock"].bottom, merge["cellblock"].left

    max_h = MAX_MERGE_ANIMATION_WINDOW_SIZE[0]
    max_w = MAX_MERGE_ANIMATION_WINDOW_SIZE[1]
    cell_h = len(img_cells[0][0])
    cell_w = len(img_cells[0][0][0])
    window_h = rows * cell_h + int(cell_h * 1.0)
    window_w = cols * cell_w + int(cell_w * 1.0)
    cellblock_h = (bottom - top + 1) * cell_h
    cellblock_w = (right - left + 1) * cell_w
    start_h = (window_h - cellblock_h) // 2
    start_w = (window_w - cellblock_w) // 2
    scale = min(max_h / window_h, max_w / window_w)

    whiteboard = np.zeros((window_h, window_w, 3), dtype=np.uint8)
    whiteboard.fill(255)

    # assemble image as specifed in cellblock blueprint
    for i, _ in enumerate(cellblock):
        for j, _ in enumerate(cellblock[0]):
            if cellblock[i][j].is_valid():
                cdata = cellblock[i][j]
                paste = img_cells[cdata.img_id][cdata.transform]
                y_offset = start_h + (i - top) * cell_h
                x_offset = start_w + (j - left) * cell_w
                whiteboard[y_offset:y_offset + cell_h, x_offset:x_offset + cell_w] = paste
                if draw_borders and cellblock[i][j].img_id == celldata.img_id:
                    if celldata.dir == 0:
                        cv2.line(whiteboard, (x_offset, y_offset + cell_h),
                                 (x_offset + cell_w, y_offset + cell_h),
                                 (0, 0, 255), int(2 / scale))
                    elif celldata.dir == 1:
                        cv2.line(whiteboard, (x_offset, y_offset),
                                 (x_offset + cell_w, y_offset),
                                 (0, 0, 255), int(2 / scale))
                    elif celldata.dir == 2:
                        cv2.line(whiteboard, (x_offset + cell_w, y_offset),
                                 (x_offset + cell_w, y_offset + cell_h),
                                 (0, 0, 255), int(2 / scale))
                    elif celldata.dir == 3:
                        cv2.line(whiteboard, (x_offset, y_offset),
                                 (x_offset, y_offset + cell_h),
                                 (0, 0, 255), int(2 / scale))

    whiteboard = cv2.resize(whiteboard, (int(window_w * scale), int(window_h * scale)))
    whiteboard = cv2.copyMakeBorder(whiteboard, 0, 100, 0, 0, cv2.BORDER_CONSTANT)
    whiteboard = cv2.putText(whiteboard,
                             text + " similarity score =" +
                             str(celldata.score), (30, len(whiteboard) - 50),
                             fontFace=cv2.LINE_AA, fontScale=0.5, color=(255, 255, 255))
    cv2.imshow("jigsaw puzzle solver", whiteboard)
    cv2.waitKey(interval_millis)


def _display_minimum_spanning_tree(merge, img_cells, rows, cols, text="", interval_millis=200):
    """
    Private method, shows remaining image fragments.
    """
    cellblock = merge["cellblock"].data
    bottom, right, top, left = merge["cellblock"].top, merge["cellblock"].right, \
                               merge["cellblock"].bottom, merge["cellblock"].left

    max_h = MAX_MERGE_ANIMATION_WINDOW_SIZE[0]
    max_w = MAX_MERGE_ANIMATION_WINDOW_SIZE[1]
    cell_h = len(img_cells[0][0])
    cell_w = len(img_cells[0][0][0])
    window_h = rows * cell_h + int(cell_h * 1.0)
    window_w = cols * cell_w + int(cell_w * 1.0)
    cellblock_h = (bottom - top + 1) * cell_h
    cellblock_w = (right - left + 1) * cell_w
    start_h = (window_h - cellblock_h) // 2
    start_w = (window_w - cellblock_w) // 2
    scale = min(max_h / window_h, max_w / window_w)

    whiteboard = np.zeros((window_h, window_w, 3), dtype=np.uint8)
    whiteboard.fill(255)

    # assemble image as specifed in cellblock blueprint
    for i, _ in enumerate(cellblock):
        for j, _ in enumerate(cellblock[0]):
            if cellblock[i][j].is_valid():
                cdata = cellblock[i][j]
                paste = img_cells[cdata.img_id][cdata.transform]
                y_offset = start_h + (i - top) * cell_h
                x_offset = start_w + (j - left) * cell_w
                whiteboard[y_offset:y_offset + cell_h, x_offset:x_offset + cell_w] = paste

    for i, _ in enumerate(cellblock):
        for j, _ in enumerate(cellblock[0]):
            if cellblock[i][j].is_valid():
                cdata = cellblock[i][j]
                y_offset = start_h + (i - top) * cell_h
                x_offset = start_w + (j - left) * cell_w
                if cdata.dir == 0:
                    cv2.line(whiteboard, (x_offset + cell_w // 2, y_offset + cell_h // 2),
                             (x_offset + cell_w // 2, y_offset + 3 * cell_h // 2),
                             (0, 0, 255), int(2 / scale))
                if cdata.dir == 1:
                    cv2.line(whiteboard, (x_offset + cell_w // 2, y_offset + cell_h // 2),
                             (x_offset + cell_w // 2, y_offset - cell_h // 2),
                             (0, 0, 255), int(2 / scale))
                if cdata.dir == 2:
                    cv2.line(whiteboard, (x_offset + cell_w // 2, y_offset + cell_h // 2),
                             (x_offset + 3 * cell_w // 2, y_offset + cell_h // 2),
                             (0, 0, 255), int(2 / scale))
                if cdata.dir == 3:
                    cv2.line(whiteboard, (x_offset + cell_w // 2, y_offset + cell_h // 2),
                             (x_offset - cell_w // 2, y_offset + cell_h // 2),
                             (0, 0, 255), int(2 / scale))

    whiteboard = cv2.resize(whiteboard, (int(window_w * scale), int(window_h * scale)))
    whiteboard = cv2.copyMakeBorder(whiteboard, 0, 100, 0, 0, cv2.BORDER_CONSTANT)
    whiteboard = cv2.putText(whiteboard,
                             text + " similarity score =" +
                             str(merge["celldata"].score), (30, len(whiteboard) - 50),
                             fontFace=cv2.LINE_AA, fontScale=0.5, color=(255, 255, 255))
    cv2.imshow("jigsaw puzzle solver", whiteboard)
    cv2.waitKey(interval_millis)
