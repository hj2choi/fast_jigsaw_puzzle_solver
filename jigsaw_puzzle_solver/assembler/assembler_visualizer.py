""" assembler_visualizer.py
assembler.py helper routines for visualizing images assembly animation.

note: cv2.imshow() works best when kept in the main thread.
"""

import cv2
import numpy as np
import threading as th

MAX_PIECES_WINDOW_SIZE = (300, 400)  # height, width
MAX_MERGE_ANIMATION_WINDOW_SIZE = (600, 1080)  # height, width


def start_assembly_animation(merge_history, img_pieces_unaligned, img_pieces,
                             show_spanning_tree=False, interval_millis=200):
    """
    animation routine.

    Args:
        merge_history (list of dict): [{blueprint, piece}]
        img_pieces_unaligned (list of cv2 images): unaligned image pieces
        img_pieces (list of cv2 images): aligned (match width & height) image pieces
        show_spanning_tree (bool): show minimum spanning tree on top of the animation.
        interval_millis (int): time between each step of the animation
    """
    merged_id_list = []

    if not merge_history:
        return
    rows, cols = merge_history[-1]["blueprint"].block_size()

    try:
        if show_spanning_tree:
            for i, merge in enumerate(merge_history):
                # longer interval for last iteration
                interval_multiplier = 1 if i < len(merge_history) - 1 else 4
                _display_remaining_pieces(img_pieces_unaligned, merged_id_list, rows, cols,
                                          merge["piece"].img_id)
                if _display_minimum_spanning_tree(merge, img_pieces, rows, cols,
                                                  str(i + 1) + "/" + str(len(merge_history)),
                                                  interval_multiplier * interval_millis):
                    break

        else:
            for i, merge in enumerate(merge_history):
                # longer interval for last iteration
                interval_multiplier = 1 if i < len(merge_history) - 1 else 4
                _display_remaining_pieces(img_pieces_unaligned, merged_id_list, rows, cols,
                                          merge["piece"].img_id)
                if _display_current_assembly_progress(merge, img_pieces, rows, cols,
                                                      str(i + 1) + "/" + str(len(merge_history)),
                                                      i < len(merge_history) - 1,
                                                      interval_multiplier * interval_millis):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def _display_remaining_pieces(img_pieces_unaligned, merged_id_list, rows, cols, merge_id):
    """
    Private method, shows remaining image pieces.
    """
    merged_id_list.append(merge_id)

    max_h = MAX_PIECES_WINDOW_SIZE[0]
    max_w = MAX_PIECES_WINDOW_SIZE[1]
    max_piece_len = max(len(img_pieces_unaligned[0][0]), len(img_pieces_unaligned[0][0][0]))
    mid_align_offset = (max_piece_len -
                        min(len(img_pieces_unaligned[0][0]), len(img_pieces_unaligned[0][0][0]))) // 2
    window_h = int((rows * max_piece_len) * 1.15)
    window_w = int((cols * max_piece_len) * 1.15)
    start_h = 20
    start_w = 20
    scale = min(max_h / window_h, max_w / window_w)

    whiteboard = np.zeros((window_h, window_w, 3), dtype=np.uint8)
    whiteboard.fill(255)
    for i, _ in enumerate(img_pieces_unaligned):
        placeholder = np.full((max_piece_len, max_piece_len, 3), 245)
        paste = img_pieces_unaligned[i][0]
        cell_h = len(paste)
        cell_w = len(paste[0])
        y_offset = start_h + (i // cols) * int(max_piece_len * 1.1)
        x_offset = start_w + (i % cols) * int(max_piece_len * 1.1)

        try:
            whiteboard[y_offset:y_offset + max_piece_len,
                        x_offset:x_offset + max_piece_len] = placeholder
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
    cv2.imshow("puzzle pieces", whiteboard)


def _display_current_assembly_progress(merge, img_pieces, rows, cols, text="", draw_borders=True,
                                       interval_millis=200):
    """
    displays current assembly progress
    Returns:
        exit flag (bool): raises flag upon esc keystroke
    """
    blueprint = merge["blueprint"].data
    piece = merge["piece"]
    bottom, right, top, left = merge["blueprint"].top, merge["blueprint"].right, \
                               merge["blueprint"].bottom, merge["blueprint"].left

    max_h = MAX_MERGE_ANIMATION_WINDOW_SIZE[0]
    max_w = MAX_MERGE_ANIMATION_WINDOW_SIZE[1]
    piece_h = len(img_pieces[0][0])
    piece_w = len(img_pieces[0][0][0])
    window_h = rows * piece_h + int(piece_h * 1.0)
    window_w = cols * piece_w + int(piece_w * 1.0)
    blueprint_h = (bottom - top + 1) * piece_h
    blueprint_w = (right - left + 1) * piece_w
    start_h = (window_h - blueprint_h) // 2
    start_w = (window_w - blueprint_w) // 2
    scale = min(max_h / window_h, max_w / window_w)

    whiteboard = np.zeros((window_h, window_w, 3), dtype=np.uint8)
    whiteboard.fill(255)

    # assemble image as specified in blueprint
    for i, _ in enumerate(blueprint):
        for j, _ in enumerate(blueprint[0]):
            if blueprint[i][j].is_valid():
                curr_piece = blueprint[i][j]
                paste = img_pieces[curr_piece.img_id][curr_piece.orientation]
                y_offset = start_h + (i - top) * piece_h
                x_offset = start_w + (j - left) * piece_w
                whiteboard[y_offset:y_offset + piece_h, x_offset:x_offset + piece_w] = paste
                if draw_borders and blueprint[i][j].img_id == piece.img_id:
                    if piece.dir == 0:
                        cv2.line(whiteboard, (x_offset, y_offset + piece_h),
                                 (x_offset + piece_w, y_offset + piece_h),
                                 (0, 0, 255), int(2 / scale))
                    elif piece.dir == 1:
                        cv2.line(whiteboard, (x_offset, y_offset),
                                 (x_offset + piece_w, y_offset),
                                 (0, 0, 255), int(2 / scale))
                    elif piece.dir == 2:
                        cv2.line(whiteboard, (x_offset + piece_w, y_offset),
                                 (x_offset + piece_w, y_offset + piece_h),
                                 (0, 0, 255), int(2 / scale))
                    elif piece.dir == 3:
                        cv2.line(whiteboard, (x_offset, y_offset),
                                 (x_offset, y_offset + piece_h),
                                 (0, 0, 255), int(2 / scale))

    whiteboard = cv2.resize(whiteboard, (int(window_w * scale), int(window_h * scale)))
    whiteboard = cv2.copyMakeBorder(whiteboard, 0, 100, 0, 0, cv2.BORDER_CONSTANT)
    whiteboard = cv2.putText(whiteboard,
                             text + " similarity score =" +
                             str(piece.score), (30, len(whiteboard) - 50),
                             fontFace=cv2.LINE_AA, fontScale=0.5, color=(255, 255, 255))
    cv2.imshow("jigsaw puzzle solver", whiteboard)
    return cv2.waitKey(interval_millis) == 27


def _display_minimum_spanning_tree(merge, img_pieces, rows, cols, text="", interval_millis=200):
    """
    displays current assembly progress with minimum spanning visualization on top.
    Returns:
        exit flag (bool): raises flag upon esc keystroke
    """
    blueprint = merge["blueprint"].data
    bottom, right, top, left = merge["blueprint"].top, merge["blueprint"].right, \
                               merge["blueprint"].bottom, merge["blueprint"].left

    max_h = MAX_MERGE_ANIMATION_WINDOW_SIZE[0]
    max_w = MAX_MERGE_ANIMATION_WINDOW_SIZE[1]
    piece_h = len(img_pieces[0][0])
    piece_w = len(img_pieces[0][0][0])
    window_h = rows * piece_h + int(piece_h * 1.0)
    window_w = cols * piece_w + int(piece_w * 1.0)
    blueprint_h = (bottom - top + 1) * piece_h
    blueprint_w = (right - left + 1) * piece_w
    start_h = (window_h - blueprint_h) // 2
    start_w = (window_w - blueprint_w) // 2
    scale = min(max_h / window_h, max_w / window_w)

    whiteboard = np.zeros((window_h, window_w, 3), dtype=np.uint8)
    whiteboard.fill(255)

    # assemble image as specified in blueprint
    for i, _ in enumerate(blueprint):
        for j, _ in enumerate(blueprint[0]):
            if blueprint[i][j].is_valid():
                piece = blueprint[i][j]
                paste = img_pieces[piece.img_id][piece.orientation]
                y_offset = start_h + (i - top) * piece_h
                x_offset = start_w + (j - left) * piece_w
                whiteboard[y_offset:y_offset + piece_h, x_offset:x_offset + piece_w] = paste

    for i, _ in enumerate(blueprint):
        for j, _ in enumerate(blueprint[0]):
            if blueprint[i][j].is_valid():
                piece = blueprint[i][j]
                y_offset = start_h + (i - top) * piece_h
                x_offset = start_w + (j - left) * piece_w
                if piece.dir == 0:
                    cv2.line(whiteboard, (x_offset + piece_w // 2, y_offset + piece_h // 2),
                             (x_offset + piece_w // 2, y_offset + 3 * piece_h // 2),
                             (0, 0, 255), int(2 / scale))
                if piece.dir == 1:
                    cv2.line(whiteboard, (x_offset + piece_w // 2, y_offset + piece_h // 2),
                             (x_offset + piece_w // 2, y_offset - piece_h // 2),
                             (0, 0, 255), int(2 / scale))
                if piece.dir == 2:
                    cv2.line(whiteboard, (x_offset + piece_w // 2, y_offset + piece_h // 2),
                             (x_offset + 3 * piece_w // 2, y_offset + piece_h // 2),
                             (0, 0, 255), int(2 / scale))
                if piece.dir == 3:
                    cv2.line(whiteboard, (x_offset + piece_w // 2, y_offset + piece_h // 2),
                             (x_offset - piece_w // 2, y_offset + piece_h // 2),
                             (0, 0, 255), int(2 / scale))

    whiteboard = cv2.resize(whiteboard, (int(window_w * scale), int(window_h * scale)))
    whiteboard = cv2.copyMakeBorder(whiteboard, 0, 100, 0, 0, cv2.BORDER_CONSTANT)
    whiteboard = cv2.putText(whiteboard,
                             text + " similarity score =" +
                             str(merge["piece"].score), (30, len(whiteboard) - 50),
                             fontFace=cv2.LINE_AA, fontScale=0.5, color=(255, 255, 255))
    cv2.imshow("jigsaw puzzle solver", whiteboard)
    return cv2.waitKey(interval_millis) == 27
