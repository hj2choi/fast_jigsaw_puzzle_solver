""" assembler.py
main jigsaw puzzle solver algorithm.

reads all image pieces with provided prefix and reconstruct pieces back to original images.
adapted Prim's Minimum Spanning Tree algorithm.

3D distance matrix is computed in parallel.
"""
import os
import time
import pickle as pkl
from multiprocessing import Pool

import cv2
import numpy as np

from . import assembler_helpers as helpers
from . import assembler_visualizer as vis
from .assembler_data_structures import PuzzlePiece, ConstructionBlueprint, LinkedHashmapPriorityQueue


# image pieces count threshold for parallel distance matrix computation
PARALLEL_COMPUTATION_MIN_IMAGES = 128
# the number of CPU cores to be used to construct similarity matrix in parallel
PROCESS_COUNT = 3


class ImageAssembler:
    """
    Image assembler class.

    Usage:
        from jigsaw_puzzle_solver.assembler import imageAssembler
        assembler = ImageAssembler.load_from_filepath(directory, prefix, max_cols, max_rows)
        assembler.assemble()
        assembler.start_assembly_animation()
        assembler.save_assembled_image()

    Attributes:
        raw_imgs (2d list of cv2 images):
            collection of puzzle pieces with all possible orientations
            (4 for square pieces, 8 for rectangle ones)

    Methods:
        assemble() -> void
        save_assembled_image() -> void
        save_assembled_image(filepath: str) -> void
        start_assembly_animation(interval_millis: int) -> void
    """

    def __init__(self, data=([], [])):
        self.raw_imgs_unaligned = data[0]  # for visualization
        self.raw_imgs = data[1]  # image fragments with all combination of orientations
        self.orientation_cnt = len(self.raw_imgs[0])  # number of possible orientation states
        self.sim_matrix = np.zeros((len(self.raw_imgs), len(self.raw_imgs),
                                    self.orientation_cnt * 4))
        self.idx_map = helpers.map4 if self.orientation_cnt == 4 else helpers.map8
        # similarity matrix depth mapper for filling out symmetric parts
        self.mat_sym_dmapper = helpers.mat_sym_dmap16 if self.orientation_cnt == 4 \
            else helpers.mat_sym_dmap32

        self.blueprint = None  # blueprint for image reconstruction
        self.merge_history = []  # assembly history in y, x positions. ex) [[0, 0], [0, 1], [-1,0], ...]

    @classmethod
    def load_from_filepath(cls, directory, prefix):
        """ Constructor method. Loads all puzzle pieces with given prefix.

        Args:
            directory (str): directory storing puzzle pieces
            prefix (str): prefix of fragmented image filenames (should be set fragment_images.py)
        """

        raw_images = []
        raw_images_unaligned = []
        for (_, _, filenames) in os.walk(directory):
            for filename in filenames:
                if filename.startswith(prefix):
                    img = cv2.imread(directory + "/" + filename)
                    # square images
                    if len(img) == len(img[0]):
                        all_orientations = []
                        for i in range(8):
                            if i == 4:
                                img = np.flip(img, 0)
                            all_orientations.append(np.copy(np.rot90(img, i)))
                        raw_images.append(all_orientations)
                        raw_images_unaligned.append(all_orientations)
                    # rectangular images: reduce 3d distance matrix by 2x.
                    else:
                        raw_images_unaligned.append([img, np.flip(img, 1),
                                                     np.flip(img, 0), np.flip(np.flip(img, 0), 1)])
                        img = np.rot90(img) if len(img) > len(img[0]) else img
                        raw_images.append([img, np.flip(img, 1), np.flip(img, 0),
                                           np.flip(np.flip(img, 0), 1)])
        return cls((raw_images_unaligned, raw_images))

    def assemble(self):
        """
            assemble puzzle pieces back to original image.
            Prim's Minimum Spanning Tree algorithm with
            Linked Hashmap implementation of the Priority Queue.
        """

        def _best_fit_piece_at(y, x):
            """
            From the list of unmerged pieces,
            find one that can be most naturally stitched at position x, y
            """
            nonlocal unused_ids
            best_candidate = PuzzlePiece()
            for img_id in unused_ids:  # for all remaining images
                for adj in self.blueprint.get_active_neighbors(y, x):  # for all adjacent images
                    for k in range(self.orientation_cnt * adj.dir,  # for all transformations
                                   self.orientation_cnt * adj.dir + self.orientation_cnt):
                        score = self.sim_matrix[adj.img_id][img_id][self.idx_map(adj.orientation, k)]
                        if best_candidate.score < score or not best_candidate.is_valid():
                            best_candidate.set(img_id, k % self.orientation_cnt, score, y, x, adj.dir)
            return best_candidate

        def _dequeue_and_merge():
            """
            Dequeue image cell from the priority queue and place it on the blueprint.
            Then, remove all duplicate image cells from the priority queue.
            """
            nonlocal p_queue, unused_ids
            piece, duplicates = p_queue.dequeue_and_remove_duplicate_ids()
            self.blueprint.activate_position(piece)
            unused_ids.remove(piece.img_id)
            print("image merged: ", piece.tostring(), "\t",
                  len(self.raw_imgs) - len(unused_ids), "/", len(self.raw_imgs), flush=True)
            self.merge_history.append(piece)
            # print("current-blueprint:\n", blueprint.data)
            return piece, duplicates

        def _enqueue_all_frontiers(frontier_pieces_list):
            """
            for all next possible puzzle piece placement positions,
            find best fit piece at each position and append them puzzle pieces to the priority queue.
            """
            nonlocal p_queue
            for frontier in frontier_pieces_list:
                if self.blueprint.validate_position(*frontier.pos()):
                    pc = _best_fit_piece_at(*frontier.pos())
                    if pc.is_valid():
                        p_queue.enqueue(pc.img_id, pc)

        # initialization.
        self._construct_similarity_matrix()
        s_time = time.time()
        self.merge_history = []  # reset merge_history
        unused_ids = [*range(0, len(self.raw_imgs))]  # remaining cells
        self.blueprint = ConstructionBlueprint(len(self.raw_imgs), len(self.raw_imgs))  # image reconstruction blueprint
        p_queue = LinkedHashmapPriorityQueue(len(self.raw_imgs))  # priority queue for MST algorithm
        p_queue.enqueue(0, PuzzlePiece(0, 0, 1.0, self.blueprint.top, self.blueprint.left))  # source node`

        # the main MST assembly algorithm loop
        while not p_queue.is_empty():
            # do not consider position that's already occupied
            if not self.blueprint.validate_position(*p_queue.peek().pos()):
                p_queue.dequeue()
                continue
            # dequeue image cell from the priority queue, and merge it towards the final image form.
            piece, duplicates = _dequeue_and_merge()
            # add best fit image cell at all frontier positions to the priority queue
            _enqueue_all_frontiers(self.blueprint.get_inactive_neighbors(*piece.pos()) + duplicates)

        print("MST assembly algorithm:", time.time() - s_time, "seconds")

    def save_assembled_image(self, filepath):
        """
            save assembled image to file
        """
        top, bottom, right, left = self.blueprint.top, self.blueprint.bottom, self.blueprint.right, self.blueprint.left
        piece_h, piece_w = len(self.raw_imgs[0][0]), len(self.raw_imgs[0][0][0])
        blueprint_h, blueprint_w = (top - bottom + 1) * piece_h, (right - left + 1) * piece_w

        whiteboard = np.zeros((blueprint_h, blueprint_w, 3), dtype=np.uint8)
        whiteboard.fill(0)
        for i in range(len(self.blueprint.data)):
            for j in range(len(self.blueprint.data)):
                piece = self.blueprint.data[i][j]
                if piece.is_valid():
                    paste = self.raw_imgs[piece.img_id][piece.orientation]
                    y_offset, x_offset = (i - bottom) * piece_h, (j - left) * piece_w
                    whiteboard[y_offset: y_offset + piece_h, x_offset: x_offset + piece_w] = paste
        cv2.imwrite(filepath + ".png", whiteboard)

    def start_assembly_animation(self, show_spanning_tree, interval_millis=200):
        """
            show animation after assembly process is complete.
        """
        vis.start_assembly_animation(self.blueprint, self.merge_history, self.raw_imgs_unaligned,
                                     self.raw_imgs, show_spanning_tree, interval_millis)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        private methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _construct_similarity_matrix(self):
        """
            construct similarity matrix for all image pairs,
            considers all combination of stitching directions and orientations.
            shape of similarity matrix = (row, col, [16 or 32])
        """

        raw_imgs_normalized = np.array(self.raw_imgs) / 256  # normalize

        s_time = time.time()
        # try parallel preprocessing for large number of images
        if not self._construct_similarity_matrix_parallel(raw_imgs_normalized, self.orientation_cnt):
            # serial processing. Only compute for the upper triangular region.
            for i in range(len(self.raw_imgs)):
                for j in range(len(self.raw_imgs)):
                    if i < j:
                        for k in range(self.orientation_cnt * 4):
                            self.sim_matrix[i][j][k] = helpers.img_borders_similarity(
                                raw_imgs_normalized[j][k % self.orientation_cnt],
                                raw_imgs_normalized[i][0], k // self.orientation_cnt)

        # fill up the missing lower triangular region of the similarity matrix.
        for i in range(len(self.raw_imgs)):
            for j in range(len(self.raw_imgs)):
                if i > j:
                    for k in range(self.orientation_cnt * 4):
                        self.sim_matrix[i][j][k] = self.sim_matrix[j][i][self.mat_sym_dmapper(k)]
        print("similarity matrix construction:", time.time() - s_time, "seconds")

    def _construct_similarity_matrix_parallel(self, raw_imgs_norm, orientation_cnt):
        """
            construct similarity matrix for all image pairs,
            considers all combination of stitching directions and orientations.
            shape of the similarity matrix = (row, col, [16 or 32])
        """
        if len(self.raw_imgs) < PARALLEL_COMPUTATION_MIN_IMAGES * 4 / orientation_cnt:
            return False
        try:
            s_time = time.time()
            with Pool(PROCESS_COUNT, self._init_process,
                      (raw_imgs_norm, orientation_cnt)) as pool:
                print("child process creation overhead:", time.time() - s_time, "seconds")
                self.sim_matrix = np.reshape(pool.map(self._compute_elementwise_similarity,
                                                      np.ndenumerate(self.sim_matrix)),
                                             (len(self.raw_imgs), len(self.raw_imgs),
                                              orientation_cnt * 4))
            return True
        except Exception as exception:
            print("Failed to start process")
            print(exception)
            return False

    @staticmethod
    def _init_process(raw_imgs_norm, _t_cnt):
        global RAW_IMGS_NORM, ORIENTATION_CNT
        RAW_IMGS_NORM = raw_imgs_norm
        ORIENTATION_CNT = _t_cnt

    @staticmethod
    def _compute_elementwise_similarity(x):
        global RAW_IMGS_NORM, ORIENTATION_CNT
        i, j, k = x[0][0], x[0][1], x[0][2]
        # only compute for the upper triangular region.
        if i > j:
            return 0
        return helpers.img_borders_similarity(
            RAW_IMGS_NORM[j][k % ORIENTATION_CNT],
            RAW_IMGS_NORM[i][0], k // ORIENTATION_CNT)
