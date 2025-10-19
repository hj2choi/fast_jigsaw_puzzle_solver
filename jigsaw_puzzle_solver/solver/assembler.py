""" assembler.py
main jigsaw puzzle solver algorithm.

Reads all image pieces with a provided filename prefix and reconstructs pieces into the original image.
Adapts Prim's Minimum Spanning Tree algorithm, and computes a 3D distance matrix in parallel.
"""
import os
import time
import pickle as pkl
from multiprocessing import Pool

import cv2
import numpy as np

from . import assembler_helpers as helpers
from . import assembler_visualizer as vis
from .assembler_data_structures import PuzzlePiece, PuzzleBlock, LinkedHashmapPriorityQueue

# Threshold for the number of images required to trigger parallel distance matrix computation.
PARALLEL_COMPUTATION_MIN_IMAGES = 128
# The number of CPU cores to be used to construct the similarity matrix in parallel
PROCESS_COUNT = 3


class ImageAssembler:
    """
    Jigsaw puzzle assembler class.

    Usage:
        from jigsaw_puzzle_solver.assembler import ImageAssembler
        assembler = ImageAssembler.load_from_filepath(directory, prefix, max_cols, max_rows)
        assembler.assemble()
        assembler.start_assembly_animation()
        assembler.save_assembled_image()

    Attributes:
        raw_imgs (2d list of cv2 images):
            collection of puzzle pieces with all possible orientations
            (4 for square pieces, 8 for rectangle ones)
        max_rows (int): optional, the maximum number of rows for the reconstructed image.
        max_cols (int): optional, max_cols, max_rows are interchangeable

    Methods:
        assemble() -> void
        save_assembled_image() -> void
        save_assembled_image(filepath: str) -> void
        start_assembly_animation(interval_millis: int) -> void
    """

    def __init__(self, data=([], []), max_rows=0, max_cols=0):
        self.max_rows, self.max_cols = max_rows, max_cols
        self.raw_imgs_unaligned = data[0]  # rectangular images with all orientations, for visualization.
        self.raw_imgs = data[1]  # rectangular images with all orientations, aligned to match width & height.
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
    def load_from_filepath(cls, directory, prefix, max_cols=0, max_rows=0):
        """ Constructor method. Loads all puzzle pieces with provided prefix.

        Args:
            directory (str): directory storing puzzle pieces
            prefix (str): prefix of puzzle piece filenames (created using create_jigsaw_pieces.py)
            max_cols (int): optional, the maximum number of rows for the reconstructed image.
            max_rows (int): optional, the maximum number of columns for the reconstructed image.
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
        return cls((raw_images_unaligned, raw_images), max_cols, max_rows)

    def assemble(self):
        """
        Assemble puzzle pieces back into the original image using Prim's Minimum Spanning Tree algorithm with
        Linked Hashmap implementation of the Priority Queue.

        Algorithm:
            1. Initialize the 3D similarity matrix between each puzzle piece.
            2. Initialize a ConstructionBlueprint object to keep track of the merged pieces.
            3. Initialize a LinkedHashmapPriorityQueue object to store the puzzle pieces that will be merged next.
            4. Add the first puzzle piece to the priority queue.
            5. Loop through the priority queue until it is empty:
                a. Dequeue the puzzle piece with the highest score from the priority queue.
                b. Add the puzzle piece to the ConstructionBlueprint object.
                c. Find the best fit puzzle pieces for all possible adjacent positions.
                d. Add the best fit puzzle pieces to the priority queue.

        Returns:
            None
        """
        
        def _best_fit_piece_at(y, x, unused_ids, blueprint):
            """
            Find the puzzle piece that can be most naturally stitched at the given position.

            Args:
                y: int, the row position.
                x: int, the column position.
                unused_ids: list of int, remaining puzzle piece IDs to consider.
                blueprint: PuzzleBlock object, current state of assembled pieces.

            Returns:
                The PuzzlePiece object that is the best fit.
            """
            best_candidate = PuzzlePiece()
            for img_id in unused_ids:  # for all remaining images
                for adj in blueprint.get_active_neighbors(y, x):  # for all adjacent images
                    for k in range(self.orientation_cnt * adj.dir,  # for all transformations
                                   self.orientation_cnt * adj.dir + self.orientation_cnt):
                        score = self.sim_matrix[adj.img_id][img_id][self.idx_map(adj.orientation, k)]
                        if best_candidate.score < score or not best_candidate.is_valid():
                            best_candidate.set(img_id, k % self.orientation_cnt, score, y, x, adj.dir)
            return best_candidate

        def _dequeue_and_merge(p_queue, unused_ids, blueprint):
            """
            Dequeue the puzzle piece with the highest score from the priority queue
            and merge it into the ConstructionBlueprint object.
            Then remove all duplicate puzzle pieces from the priority queue.

            Args:
                p_queue: LinkedHashmapPriorityQueue object, queue of pieces to be merged.
                unused_ids: list of int, remaining puzzle piece IDs (will be modified).
                blueprint: PuzzleBlock object, current state of assembled pieces.

            Returns:
                The PuzzlePiece object that was dequeued and merged.
            """
            piece, duplicates = p_queue.dequeue_and_remove_duplicate_ids()
            blueprint.activate_position(piece)
            unused_ids.remove(piece.img_id)
            print("image merged: ", piece.tostring(), "\t",
                  len(self.raw_imgs) - len(unused_ids), "/", len(self.raw_imgs), flush=True)
            self.merge_history.append(piece)
            # print("current-blueprint:\n", blueprint.data)
            return piece, duplicates

        def _enqueue_all_frontiers(frontier_pieces_list, p_queue, unused_ids, blueprint):
            """
            For all next possible puzzle piece placement positions,
            find the best fit piece at each position and append them puzzle pieces to the priority queue.

            Args:
                frontier_pieces_list: List of PuzzlePiece objects representing the positions of the puzzle pieces on the
                    frontier of the ConstructionBlueprint.
                p_queue: LinkedHashmapPriorityQueue object, queue of pieces to be merged.
                unused_ids: list of int, remaining puzzle piece IDs to consider.
                blueprint: PuzzleBlock object, current state of assembled pieces.

            Returns:
                None
            """
            for frontier in frontier_pieces_list:
                if blueprint.validate_position(*frontier.pos()):
                    pc = _best_fit_piece_at(*frontier.pos(), unused_ids, blueprint)
                    if pc.is_valid():
                        p_queue.enqueue(pc.img_id, pc)

        # initialization.
        self._compute_similarity_matrix()
        s_time = time.time()
        self.merge_history = []
        unused_ids = [*range(0, len(self.raw_imgs))]  # remaining pieces
        self.blueprint = PuzzleBlock(self.max_rows if self.max_rows > 0 else len(self.raw_imgs),
                                     self.max_cols if self.max_cols > 0 else len(self.raw_imgs))
        p_queue = LinkedHashmapPriorityQueue(len(self.raw_imgs))

        # add the first puzzle piece to the priority queue
        p_queue.enqueue(0, PuzzlePiece(0, 0, 1.0, self.blueprint.top, self.blueprint.left))

        # MST assembly algorithm loop
        while not p_queue.is_empty():
            # do not consider a position that's already activated by the blueprint
            if not self.blueprint.validate_position(*p_queue.peek().pos()):
                p_queue.dequeue()
                continue
            # dequeue puzzle piece from the priority queue, and merge it towards the final image form.
            piece, duplicates = _dequeue_and_merge(p_queue, unused_ids, self.blueprint)
            # add the best fit puzzle pieces at all frontier positions to the priority queue
            _enqueue_all_frontiers(self.blueprint.get_inactive_neighbors(*piece.pos()) + duplicates, 
                                 p_queue, unused_ids, self.blueprint)
        print("MST assembly algorithm:", time.time() - s_time, "seconds")

    def save_assembled_image(self, filepath):
        """
        save the reconstructed image to a file

        Args:
            filepath (str): The path and filename to save the image to.
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
        Show an animation of the assembly process after it is complete.

        Args:
            show_spanning_tree (bool): if True, the animation will display the MST used during assembly.
            interval_millis (int): The interval (in milliseconds) between animation frames.
        """
        vis.start_assembly_animation(self.blueprint, self.merge_history, self.raw_imgs_unaligned,
                                     self.raw_imgs, show_spanning_tree, interval_millis)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        private methods
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def _compute_similarity_matrix(self):
        """
            Computes the similarity matrix for all image pairs by considering all
            possible combinations of stitching directions and orientations. The shape
            of the similarity matrix is (rows, cols, [16 or 32]).
        """

        # normalize the raw images.
        raw_imgs_normalized = np.array(self.raw_imgs) / 256

        s_time = time.time()
        # try parallel preprocessing for a large number of images
        if not self._compute_similarity_matrix_parallel(raw_imgs_normalized, self.orientation_cnt):
            # compute in serial processing as default. Only compute for the upper triangular region.
            for i in range(len(self.raw_imgs)):
                for j in range(len(self.raw_imgs)):
                    if i < j:
                        for k in range(self.orientation_cnt * 4):
                            # compute the similarity between the borders of the images.
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

    def _compute_similarity_matrix_parallel(self, raw_imgs_norm, orientation_cnt):
        """
            Constructs the similarity matrix for all image pairs by considering all
            possible combinations of stitching directions and orientations.
            The shape of the similarity matrix is (rows, cols, [16 or 32]). This function uses
            parallel processing if there are enough images.

            Args:
                raw_imgs_norm: The normalized raw images.
                orientation_cnt: The number of possible orientations for each image.

            Returns:
                True if the similarity matrix was constructed using parallel processing;
                False otherwise.
        """
        # check if there are enough images to justify using parallel processing.
        if len(self.raw_imgs) < PARALLEL_COMPUTATION_MIN_IMAGES * 4 / orientation_cnt:
            return False
        try:
            s_time = time.time()
            # create a process pool and use it to compute the similarity matrix in parallel
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
        """
            Initialize global variables for parallel processing.
        """
        global RAW_IMGS_NORM, ORIENTATION_CNT
        RAW_IMGS_NORM = raw_imgs_norm
        ORIENTATION_CNT = _t_cnt

    @staticmethod
    def _compute_elementwise_similarity(x):
        """
            Compute similarity between two images for a specific combination of orientations and stitching directions.
        """
        i, j, k = x[0][0], x[0][1], x[0][2]
        # only compute for the upper triangular region.
        if i > j:
            return 0
        return helpers.img_borders_similarity(
            RAW_IMGS_NORM[j][k % ORIENTATION_CNT],
            RAW_IMGS_NORM[i][0], k // ORIENTATION_CNT)
