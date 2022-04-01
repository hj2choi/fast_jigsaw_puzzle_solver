"""assemble fragmented image cells

"""

import os
import time
import copy
from multiprocessing import Pool

import cv2
import numpy as np

from . import assembler_helpers as helpers
from . import assembler_visualizer as vis
from .assembler_data_structures import CellData, CellBlock, LHashmapPriorityQueue
#import jigsaw_puzzle_solver.assembler.assembler_helpers as helpers
#import jigsaw_puzzle_solver.assembler.assembler_visualizer as vis
#from jigsaw_puzzle_solver.assembler.assembler_data_structures import CellData, CellBlock, LHashmapPriorityQueue

COMPUTE_PARALLEL_FRAGMENTS_THRESHOLD = 128  # compute similarity matrix in parallel for number of images above threshold
MAX_PROCESS_COUNT = 3  # the number of CPUs to be used to construct similarity matrix


class ImageAssembler:
    """
    Image assembler class.

    Usage:
        from jigsaw_puzzle_solver.assembler import imageAssembler
        assembler = ImageAssembler.load_from_filepath(directory, prefix, max_cols, max_rows)
        assembler.assemble()
        assembler.start_assemble_animation()
        assembler.save_assembled_image()

    Attributes:
        max_rows (int)
        max_cols (int)
            max_cols, max_rows are interchangeable
        raw_imgs (2d list of cv2 images):
            collection of image fragments with all possible orientations (4 for square fragments, 8 for rectangle ones)

    Methods:
        assemble() -> void
        save_assembled_image() -> void
        save_assembled_image(filepath: str) -> void
        start_assemble_animation(interval_millis: int) -> void
    """

    def __init__(self, data=([], []), max_cols=0, max_rows=0):
        self.max_rows, self.max_cols = max_rows, max_cols  # as specified in args
        self.raw_imgs_unaligned = data[0]  # for visualization
        self.raw_imgs = data[1]  # image fragments with all combination of transformations
        self.transforms_cnt = len(self.raw_imgs[0])  # number of possible transformation states
        self.sim_matrix = np.zeros((len(self.raw_imgs), len(self.raw_imgs), self.transforms_cnt * 4))
        self.idxmap = helpers.map4 if self.transforms_cnt == 4 else helpers.map8  # similarity matrix index mapper
        # similarity matrix depth mapper for filling out symmetric parts
        self.mat_sym_dmapper = helpers.mat_sym_dmap16 if self.transforms_cnt == 4 else helpers.mat_sym_dmap32
        self.merge_history = []  # merge history from first merge to last

    @classmethod
    def load_from_filepath(cls, directory, prefix, max_cols, max_rows):
        """ Constructor method. load all fragmented images with given prefix.

        Args:
            directory (str): directory storing fragmented images
            prefix (str): prefix of fragmented image filenames (should be set fragment_images.py)
            max_cols (int): columns constraint for image assembly
            max_rows (int): rows constraint for image assembly (max_cols, max_rows are interchangeable)
        """

        raw_images = []
        raw_images_unaligned = []
        for (_, _, filenames) in os.walk(directory):
            for filename in filenames:
                if filename.startswith(prefix):
                    img = cv2.imread(directory + "/" + filename)
                    # square images
                    if len(img) == len(img[0]):
                        all_transformations = []
                        for i in range(8):
                            if i == 4:
                                img = np.flip(img, 0)
                            all_transformations.append(np.copy(np.rot90(img, i)))
                        raw_images.append(all_transformations)
                        raw_images_unaligned.append(all_transformations)
                    # rectangle images
                    else:
                        raw_images_unaligned.append([img, np.flip(img, 1),
                                                     np.flip(img, 0), np.flip(np.flip(img, 0), 1)])
                        img = np.rot90(img) if len(img) > len(img[0]) else img
                        raw_images.append([img, np.flip(img, 1), np.flip(img, 0),
                                           np.flip(np.flip(img, 0), 1)])
        return cls((raw_images_unaligned, raw_images), max_cols, max_rows)

    def assemble(self):
        """
            assemble fragmented image cells back to original image.
            Prim's Minimum Spanning Tree algorithm with Linked Hashmap implementation of Priority Queue.
        """

        def _best_fit_cell_at(y, x):
            # from the list of unmerged image cells, find one that can be most naturally stitched at position x, y
            nonlocal cellblock, unused_ids
            t_cnt = self.transforms_cnt
            best_celldata = CellData()
            for id in unused_ids:  # for all remaining images
                for adj in cellblock.active_neighbors(y, x):  # for all adjacent images
                    for k in range(t_cnt * adj.dir, t_cnt * adj.dir + t_cnt):  # for all transformations
                        score = self.sim_matrix[adj.id][id][self.idxmap(adj.transform, k)]
                        if best_celldata.score < score or not best_celldata.is_valid():
                            best_celldata.set(id, k % t_cnt, score, y, x, adj.dir)
            return best_celldata

        def _dequeue_and_merge():
            # dequeue image cell from the priority queue and place it on the cellblock.
            # Then, remove all duplicate image cells from the priority queue.
            nonlocal p_queue, cellblock, unused_ids
            cdata, duplicates = p_queue.dequeue_and_remove_duplicate_ids()
            cellblock.activate_cell(cdata)
            unused_ids.remove(cdata.id)
            print("image merged: ", cdata.tostring(), "\t",
                  len(self.raw_imgs) - len(unused_ids), "/", len(self.raw_imgs), flush=True)
            self.merge_history.append({"cellblock": copy.deepcopy(cellblock), "celldata": copy.deepcopy(cdata)})
            # print("current-cellblock:\n", cellblock.data)
            return cdata, duplicates

        def _enqueue_all_frontiers(frontier_cells_list):
            # for all next possible image cell placement positions, find best fit cell and append to the priority queue.
            nonlocal p_queue, cellblock
            for frontier in frontier_cells_list:
                if cellblock.validate_pos(*frontier.pos()):
                    cdata = _best_fit_cell_at(*frontier.pos())
                    if cdata.is_valid():
                        p_queue.enqueue(cdata.id, cdata)

        # initialization.
        self._construct_similarity_matrix()
        s_time = time.time()
        self.merge_history = []  # reset merge_history
        unused_ids = [*range(0, len(self.raw_imgs))]  # remaining cells
        cellblock = CellBlock(self.max_rows, self.max_cols)  # blueprint for image reconstruction
        p_queue = LHashmapPriorityQueue(len(self.raw_imgs))  # priority queue for MST algorithm
        p_queue.enqueue(0, CellData(0, 0, 1.0, cellblock._hs, cellblock._ws))  # source node

        # the main MST assembly algorithm loop
        while not p_queue.is_empty():
            # do not consider position that's already used up.
            if not cellblock.validate_pos(*p_queue.peek().pos()):
                p_queue.dequeue()
                continue
            # dequeue image cell from the priority queue, and merge it towards the final image form.
            cell, duplicates = _dequeue_and_merge()
            # add best fit image cell at all frontier positions to the priority queue
            _enqueue_all_frontiers(cellblock.inactive_neighbors(*cell.pos()) + duplicates)

        print("MST assembly algorithm:", time.time() - s_time, "seconds")

    def save_assembled_image(self, filepath):
        """
            save assembled image to file
        """

        cellblock = self.merge_history[-1]["cellblock"]
        rt, ct, rs, cs = cellblock._ht, cellblock._wt, cellblock._hs, cellblock._ws
        cell_h, cell_w = len(self.raw_imgs[0][0]), len(self.raw_imgs[0][0][0])
        cellblock_h, cellblock_w = (rt - rs + 1) * cell_h, (ct - cs + 1) * cell_w

        whiteboard = np.zeros((cellblock_h, cellblock_w, 3), dtype=np.uint8)
        whiteboard.fill(0)
        for i in range(cellblock._data_len):
            for j in range(cellblock._data_len):
                celldata = cellblock._data[i][j]
                if celldata.is_valid():
                    paste = self.raw_imgs[celldata.id][celldata.transform]
                    y_offset, x_offset = (i - rs) * cell_h, (j - cs) * cell_w
                    whiteboard[y_offset: y_offset + cell_h, x_offset: x_offset + cell_w] = paste
        cv2.imwrite(filepath + ".png", whiteboard)

    def start_assemble_animation(self, interval_millis=100):
        vis.start_assemble_animation(self.merge_history, self.raw_imgs_unaligned,
                                     self.raw_imgs, interval_millis)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        private methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _construct_similarity_matrix(self):
        """
            construct similarity matrix for all image pairs,
            considers all combination of stitching directions and orientations.
            shape of similarity matrix = (row, col, [16 or 32])
        """

        t_cnt = self.transforms_cnt
        raw_imgs_norm = np.array(self.raw_imgs) / 256  # normalize

        s_time = time.time()
        # try parallel preprocessing for large number of images
        if not self._construct_similarity_matrix_parallel(raw_imgs_norm, t_cnt):
            # serial processing. Only compute for the upper triangular region.
            for i in range(len(self.raw_imgs)):
                for j in range(len(self.raw_imgs)):
                    if i < j:
                        for k in range(t_cnt * 4):
                            self.sim_matrix[i][j][k] = helpers.img_borders_similarity(
                                raw_imgs_norm[j][k % t_cnt],
                                raw_imgs_norm[i][0], k // t_cnt)

        # fill up the missing lower triangular region of the similarity matrix.
        for i in range(len(self.raw_imgs)):
            for j in range(len(self.raw_imgs)):
                if i > j:
                    for k in range(t_cnt * 4):
                        self.sim_matrix[i][j][k] = self.sim_matrix[j][i][self.mat_sym_dmapper(k)]
        print("preprocessing:", time.time() - s_time, "seconds")

    def _construct_similarity_matrix_parallel(self, raw_imgs_norm, t_cnt):
        """
            construct similarity matrix for all image pairs,
            considers all combination of stitching directions and orientations.
            shape of similarity matrix = (row, col, [16 or 32])
        """
        if len(self.raw_imgs) < COMPUTE_PARALLEL_FRAGMENTS_THRESHOLD * 4 / t_cnt:
            return False
        try:
            s_time = time.time()
            with Pool(MAX_PROCESS_COUNT, self._init_process, (np.array(raw_imgs_norm), t_cnt)) as pool:
                print("child process creation overhead:", time.time() - s_time, "seconds")
                s_time = time.time()
                self.sim_matrix = np.reshape(pool.map(self._compute_elementwise_similarity,
                                                      np.ndenumerate(np.copy(self.sim_matrix))),
                                             (len(self.raw_imgs), len(self.raw_imgs), t_cnt * 4))
            return True
        except Exception as e:
            print("Failed to start process")
            print(e)
            return False

    @staticmethod
    def _init_process(raw_imgs_norm, _t_cnt):
        global raw_images_norm_g, t_cnt
        raw_images_norm_g = raw_imgs_norm
        t_cnt = _t_cnt

    @staticmethod
    def _compute_elementwise_similarity(x):
        i, j, k = x[0][0], x[0][1], x[0][2]
        # only compute for the upper triangular.
        if i > j:
            return 0
        return helpers.img_borders_similarity(raw_images_norm_g[j][k % t_cnt], raw_images_norm_g[i][0], k // t_cnt)
