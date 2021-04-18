import sys, os, time, threading, multiprocessing
from multiprocessing import Pool
import numpy as np
import cv2
from . import img_operations as im_op
from . import merge_utils as mr
from . import visualization_utils as vis

class ImageMerger:
    """
    ImageMerger constructor. To be used with loadfromfilepath()

    @usage
    mr = mr.ImageMerger.loadfromfilepath(args, ...)
    """
    def __init__(self, data = [], cols = 0, rows = 0):
        self.rows, self.cols = rows, cols # as specified in args
        self.raw_imgs = data # image fragments with all combination of transformations
        self.transformations_count = len(self.raw_imgs[0]) # number of possible transformation states
        self.sim_matrix = np.zeros((len(self.raw_imgs), len(self.raw_imgs), self.transformations_count * 4))
        self.idxmap = mr.map4 if self.transformations_count == 4 else mr.map8  # similarity matrix index mapper
        self.mat_sym_dmap = mr.mat_sym_dmap16 if self.transformations_count == 4 else mr.mat_sym_dmap32  # matrix symmetry depth mapper
        self.merge_history = [] # merge history from first merge to last

    """
    load data from given program arguments
    """
    @classmethod
    def loadfromfilepath(cls, dir, prefix, cols, rows):
        raw_imgs = []
        for (dirpath, dirnames, filenames) in os.walk(dir):
            for filename in filenames:
                if filename.startswith(prefix):
                    img = cv2.imread(dir + "/" + filename)
                    if (len(img) == len(img[0])): # square images
                        all_transformations = []
                        for i in range(8):
                            if i == 4: img = np.flip(img, 0)
                            all_transformations.append(np.copy(np.rot90(img, i)))
                        raw_imgs.append(all_transformations)
                    else: # rectangle images
                        img = np.rot90(img) if len(img) > len(img[0]) else img
                        raw_imgs.append([img, np.flip(img, 1), np.flip(img, 0), np.flip(np.flip(img, 0), 1)])
        return cls(raw_imgs, cols, rows)

    """
    merge fragmented image cells back to original image.
    Modified Prim's MST algorithm. implemented Priority Queue with Linked Hashmap
    """
    def merge(self):
        def _best_fit_cell_at(i, j):
            t_cnt = self.transformations_count
            best_celldata = mr.CellData()
            for id in unused_ids: # for all remaining images
                for adj in cellblock.active_neighbors(i, j): # for all adjacent images
                    for k in range(t_cnt * adj.dir, t_cnt * adj.dir + t_cnt): # for all transformations
                        score = self.sim_matrix[adj.id][id][self.idxmap(adj.transform, k)]
                        if (best_celldata.score < score or not best_celldata.is_valid()):
                            best_celldata.set(id, k % t_cnt, score, i, j, adj.dir)
                            if mr.OPTIMIZE_STOP_SEARCH_THRESHOLD < score:
                                return best_celldata
            return best_celldata

        def _dequeue_and_merge():
            cdata, duplicates = p_queue.dequeue_and_remove_duplicate_ids()
            cellblock.activate_cell(cdata)
            unused_ids.remove(cdata.id)
            print("image merged: ", cdata.tostring(), "\t", len(self.raw_imgs)-len(unused_ids), "/", len(self.raw_imgs))
            self.merge_history.append({"cellblock": mr.CellBlock.copy(cellblock), "celldata": cdata})
            #print("current-cellblock:\n", cellblock.data)
            return cdata, duplicates

        def _enqueue_all_frontiers(frontier_cells_list):
            for frontier in frontier_cells_list:
                if cellblock.validate_pos(*frontier.pos()):
                    cdata = _best_fit_cell_at(*frontier.pos())
                    if cdata.is_valid(): p_queue.enqueue(cdata.id, cdata)

        # initialization
        self._construct_similaritymatrix()
        s_time = time.time()
        self.merge_history = [] # reset merge_history
        unused_ids = [*range(0, len(self.raw_imgs))] # remaining cells
        cellblock = mr.CellBlock(self.rows, self.cols) # blueprint for image reconstruction
        p_queue = mr.LHashmapPriorityQueue(len(self.raw_imgs)) # priority queue for MST algorithm
        p_queue.enqueue(0, mr.CellData(0, 0, 1.0, cellblock.hs, cellblock.ws)) # source node

        # The main loop
        while not p_queue.is_empty():
            if cellblock.validate_pos(*p_queue.peek().pos()):
                cell, duplicates = _dequeue_and_merge()
                _enqueue_all_frontiers(cellblock.inactive_neighbors(*cell.pos()) + duplicates)
            else:
                p_queue.dequeue() # throw away any invalid cells
        print("MST merge algorithm:", time.time() - s_time, "seconds")

    """
    save merged image to file
    """
    def save_merged_image(self, filepath):
        cellblock = self.merge_history[-1]["cellblock"]
        rt, ct, rs, cs = cellblock.ht, cellblock.wt, cellblock.hs, cellblock.ws
        cell_h, cell_w = len(self.raw_imgs[0][0]), len(self.raw_imgs[0][0][0])
        cellblock_h, cellblock_w = (rt - rs + 1) * cell_h, (ct - cs + 1) * cell_w

        whiteboard = np.zeros((cellblock_h, cellblock_w, 3), dtype = np.uint8)
        whiteboard.fill(0)
        for i in range(cellblock.length):
            for j in range(cellblock.length):
                celldata = cellblock.data[i][j]
                if (celldata.is_valid()):
                    paste = self.raw_imgs[celldata.id][celldata.transform]
                    y_offset, x_offset = (i - rs) * cell_h, (j - cs) * cell_w
                    whiteboard[y_offset: y_offset + cell_h, x_offset: x_offset + cell_w] = paste
        cv2.imwrite(filepath + ".png", whiteboard)

    def start_merge_animation(self, interval):
        vis.start_merge_animation(self.merge_history, self.raw_imgs, self.rows, self.cols, interval)

    """===========Private Methods============"""
    """
    construct similarity matrix for all image pairs,
    considers all combination of stitching directions and orientations.
    shape of similarity matrix = (row, col, [16 or 32])
    """
    def _construct_similaritymatrix(self):
        t_cnt = self.transformations_count
        raw_imgs_norm = np.array(self.raw_imgs) / 256 #normalize

        s_time = time.time()
        # try parallel preprocessing for large number of images
        if not self._construct_similaritymatrix_parallel(raw_imgs_norm, t_cnt):
            # serial processing. Only compute for the upper triangular.
            for i in range(len(self.raw_imgs)):
                for j in range(len(self.raw_imgs)):
                    if (i < j):
                        for k in range(t_cnt * 4):
                            self.sim_matrix[i][j][k] = im_op.img_borders_similarity(
                                                        raw_imgs_norm[j][k % t_cnt],
                                                        raw_imgs_norm[i][0], k // t_cnt)
        # fill up the missing lower triangular of the similarity matrix.
        for i in range(len(self.raw_imgs)):
            for j in range(len(self.raw_imgs)):
                if (i > j):
                    for k in range(t_cnt * 4):
                        self.sim_matrix[i][j][k] = self.sim_matrix[j][i][self.mat_sym_dmap(k)]
        print("preprocessing:", time.time() - s_time, "seconds")

    def _construct_similaritymatrix_parallel(self, raw_imgs_norm, t_cnt):
        if len(self.raw_imgs) < mr.SWITCH_TO_PARALLEL_THRESHOLD * 4 / t_cnt:
            return False
        try:
            s_time = time.time()
            with Pool(mr.MAX_PROCESS_COUNT, self._init_process, (np.array(raw_imgs_norm), t_cnt)) as pool:
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
        return False

    def _init_process(self, raw_imgs_norm, _t_cnt):
        global raw_imgs_norm_g, t_cnt
        raw_imgs_norm_g = raw_imgs_norm
        t_cnt = _t_cnt

    def _compute_elementwise_similarity(self, x):
        i, j, k = x[0][0], x[0][1], x[0][2]
        # only compute for the upper triangular.
        if (i > j):
            return 0
        return im_op.img_borders_similarity(raw_imgs_norm_g[j][k % t_cnt],
                                raw_imgs_norm_g[i][0], k // t_cnt)
