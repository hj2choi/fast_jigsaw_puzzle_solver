import sys, os, time, threading, multiprocessing
from multiprocessing import Pool
import numpy as np
import cv2
from . import img_operations as im_op
from . import merge_optimizer_utils as opt
from . import visualization_utils as vis

#direction ENUM
DIR = {
    'd': 0,
    'u': 1,
    'r': 2,
    'l': 3,
}

class ImageMerger:
    """
    ImageMerger constructor. To be used with loadfromfilepath()

    @usage
    mr = mr.ImageMerger.loadfromfilepath(args, ...)
    """
    def __init__(self, data = [], cols = 0, rows = 0):
        self.img_cells = data # image segments with all combination of transformations
        self.t_count = len(self.img_cells[0]) # number of possible transformation states
        self.cell_id_queue = np.arange(0,len(self.img_cells))# list of segments to be processed
        self.merge_cellblock = (np.zeros((max(rows, cols)*2,max(rows, cols)*2,2))-1).astype(int)  # blueprint for image reconstruction
        self.cols = cols # as specified in args
        self.rows = rows # as specified in args
        if (self.t_count == 8):
            self.sim_matrix = np.zeros((len(self.img_cells), len(self.img_cells), 32))
            self.map = opt.map8
        else:
            self.sim_matrix = np.zeros((len(self.img_cells), len(self.img_cells), 16))
            self.map = opt.map4 # similarity matrix index mapper (flip, mirror)
        self.merged_image = [] # final reassembled image
        self.merge_history = [] # used for animation

    """
    load data from given program arguments
    """
    @classmethod
    def loadfromfilepath(cls, dir, prefix, cols, rows):
        img_cells = []
        for (dirpath, dirnames, filenames) in os.walk(dir):
            for filename in filenames:
                if filename.startswith(prefix):
                    img = cv2.imread(dir+"/"+filename)
                    if (len(img) == len(img[0])):
                        all_transformations = []
                        for i in range(8):
                            if i == 4:
                                img = np.flip(img, 0)
                            all_transformations.append(np.copy(np.rot90(img, i)))
                        img_cells.append(all_transformations)
                    else:
                        img = np.rot90(img) if len(img) > len(img[0]) else img
                        img_cells.append([img,
                                        np.flip(img, 1),
                                        np.flip(img, 0),
                                        np.flip(np.flip(img, 0), 1)
                                        ])
        return cls(img_cells, cols, rows)

    """
    save merged image into a file
    """
    def save_merged_image(self, filepath):
        # assemble image as specifed in cellblock blueprint
        t_cnt = self.t_count
        self.merged_image = []
        for i in range(len(self.merge_cellblock)):
            line = []
            for j in range(len(self.merge_cellblock[0])):
                if (self.merge_cellblock[i][j][0] > -1):
                    idx = self.merge_cellblock[i][j]
                    if (len(line) == 0):
                        line = np.array(self.img_cells[idx[0]][idx[1] % t_cnt])
                    else:
                        line = np.concatenate((line, self.img_cells[idx[0]][idx[1] % t_cnt]), axis = 1)
            if (len(self.merged_image) == 0 and len(line) != 0):
                self.merged_image = np.array(line)
            elif len(line) != 0:
                self.merged_image = np.concatenate((self.merged_image, line))

        if len(self.merged_image) == 0:
            print("ERROR: MERGED IMAGE DO NOT EXIST.")
            return
        cv2.imwrite(filepath+".png", self.merged_image)

    def start_merge_process_animation(self, interval):
        vis.start_merge_process_animation(self.merge_history, self.img_cells, self.rows, self.cols, self.t_count, interval_millis = interval)


    """
    construct similarity matrix for merge algorithm.
    records similarity for all image pairs, while considering all combination of stitching directions and orientations.
    shape of similarity matrix = (row, col, 16~32)
    """
    def construct_similaritymatrix(self):
        t_cnt = self.t_count
        epoch = t_cnt * 4
        img_cells_norm = np.array(self.img_cells)/256
        # try running in parallel
        if (self.construct_similaritymatrix_parallel(img_cells_norm, t_cnt)):
            return

        s_time = time.time()
        for i in range(len(self.img_cells)):
            for j in range(len(self.img_cells)):
                if (i == j):
                    continue
                for k in range(len(self.sim_matrix[i][j])):
                    sim = im_op.img_borders_similarity(img_cells_norm[j][(k%epoch)%t_cnt],
                                            img_cells_norm[i][k//epoch], (k%epoch)//t_cnt)
                    self.sim_matrix[i][j][k] = sim
        print("preprocessing:",time.time()-s_time,"seconds")

    """
    construct similarity matrix in parallel
    """
    def construct_similaritymatrix_parallel(self, img_cells_norm, t_count):
        matrix_depth = len(self.sim_matrix[0][0])
        if len(self.img_cells) < opt.SWITCH_TO_PARALLEL_THRESHOLD * 4 / t_count:
            return False
        try:
            s_time = time.time()
            # create child process
            with Pool(opt.MAX_PROCESS_COUNT, opt.init_process, (np.array(img_cells_norm),t_count)) as pool:
                print("\parallellized preprocessing I/O overhead:",time.time()-s_time,"seconds")
                s_time = time.time()
                # map every element in similarity matrix
                self.sim_matrix = np.reshape(pool.map(opt.elementwise_similarity_parallel,
                                                    np.ndenumerate(np.copy(self.sim_matrix))),
                                                    (len(self.img_cells), len(self.img_cells),matrix_depth))
            print("parallellized preprocessing:",time.time()-s_time,"seconds")
            return True
        except Exception as e:
            print("Failed to start process")
            print(e)
            return False
        return False

    """
    finds all adjacent active cells. (active cell: cell with id >= 0)

    @Parameters
    cblock (npArray):    cellblock used for merging process (row, col)
    i (int):                row location of target cell
    j (int):                col location of target cell

    @Returns
    adj_dirs (npArray):     list of all directions to adjacent active cell [d, u, l, r]
    """
    def adjacent_active_cells(self, cblock, i, j):
        adj = []
        if (i+1 < len(cblock) and cblock[i+1][j][0] >= 0):
            adj.append({"dir":DIR['d'], "id":cblock[i+1][j][0], "transform":cblock[i+1][j][1]})
        if (i-1 > 0 and cblock[i-1][j][0] >= 0):
            adj.append({"dir":DIR['u'], "id":cblock[i-1][j][0], "transform":cblock[i-1][j][1]})
        if (j+1 < len(cblock[0]) and cblock[i][j+1][0] >= 0):
            adj.append({"dir":DIR['r'], "id":cblock[i][j+1][0], "transform":cblock[i][j+1][1]})
        if (j-1 > 0 and cblock[i][j-1][0] >= 0):
            adj.append({"dir":DIR['l'], "id":cblock[i][j-1][0], "transform":cblock[i][j-1][1]})
        return adj

    """
    finds single adjacent active cell.
    """
    def adjacent_active_cell(self, cblock, i, j):
        if (i+1 < len(cblock) and cblock[i+1][j][0] >= 0):
            return {"dir":DIR['d'], "id":cblock[i+1][j][0], "transform":cblock[i+1][j][1]}
        if (i-1 > 0 and cblock[i-1][j][0] >= 0):
            return {"dir":DIR['u'], "id":cblock[i-1][j][0], "transform":cblock[i-1][j][1]}
        if (j+1 < len(cblock[0]) and cblock[i][j+1][0] >= 0):
            return {"dir":DIR['r'], "id":cblock[i][j+1][0], "transform":cblock[i][j+1][1]}
        if (j-1 > 0 and cblock[i][j-1][0] >= 0):
            return {"dir":DIR['l'], "id":cblock[i][j-1][0], "transform":cblock[i][j-1][1]}
        return None

    """
    validate whether cellblock size is within bounds.

    @Parameters
    cellblock (npArray):    cellblock used for merging process (row, col)
    rows (int):             maximum cellblock height
    cols (int):             maximum cellblock width

    @Returns
    validated (bool):
    """
    def validate_cellblock(self, cellblock, rows = -1, cols = -1):
        rows = self.rows if rows == -1 else rows
        cols = self.cols if cols == -1 else cols
        rt, ct, rs, cs = 0, 0, rows, cols
        for i in range(len(cellblock)):
            for j in range(len(cellblock[0])):
                if cellblock[i][j][0] > -1:
                    if i < rs: rs = i
                    if i > rt: rt = i
                    if j < cs: cs = j
                    if j > ct: ct = j
        return (rt-rs < rows and ct-cs < cols) or (rt-rs < cols and ct-cs < rows)

    """
    cleanup and update cache
    """
    def clean_cellblock_cache(self, cache, i, j, removed_id):
        if j+1 < len(cache[0]): cache[i][j+1] = None
        if i+1 < len(cache): cache[i+1][j] = None
        if j-1 > 0: cache[i][j-1] = None
        if i-1 > 0: cache[i-1][j] = None
        for i in range(len(cache)):
            for j in range(len(cache[0])):
                if cache[i][j] != None and cache[i][j]["id"] == removed_id:
                    cache[i][j] = None

    """
    find best-fit image at given position
    """
    def best_fit_image_at(self, i, j, cellblock_temp):
        t_cnt = self.t_count

        adj_list = self.adjacent_active_cells(self.merge_cellblock, i, j)
        local_best_merge = {"next_cblock": cellblock_temp, "id":-1, "transform": 0, "score": -np.inf, "pos": (-1,-1)}
        for id in self.cell_id_queue: # for all images in queue
            for adj in adj_list: # for all adjacent images
                for k in range(t_cnt*adj["dir"], t_cnt*adj["dir"]+t_cnt): # for all transformations
                    score = self.sim_matrix[adj["id"]][id][self.map(adj["transform"], k)]

                    if (local_best_merge["id"]==-1 or local_best_merge["score"]<score):
                        cellblock_temp[i][j] = [id, k%t_cnt]
                        local_best_merge = {"next_cblock":cellblock_temp,
                                        "id":id, "transform":k%t_cnt, "score":score, "pos": (i,j)}
        return local_best_merge

    """
    merge image cells back to original image.

    1. place all images in a queue.
    2. start with a empty 'board' and paste random image at (r=0, c=0)
    2. while queue is not empty:
        2-1. for all possible position for any image to be pasted:
            2-1-1. find best-fit image, and its transformation and location
                    in accordance to similarity matrix
            2-1-2. save local best-fit image to cache
        2-2. paste best-fit image to the 'board'
        2-3. remove image from queue
    3. construct final image

    cache 이용한 최적화. (속도 최대 2x 상승)
    이미 최대 similarity값을 찾은 위치에 대해서는
    캐시에서 불러와서 반복적인 연산 최소화.

    @Requirements:
    sim_matrix (npArray):   similarity matrix for all image pairs.
                            similarity for all possible transformations and stitching directions should be pre-computed
    """
    def merge(self):
        self.construct_similaritymatrix()
        s_time = time.time()
        t_cnt = self.t_count
        self.merge_cellblock[self.rows][self.cols].fill(int(0))
        self.cell_id_queue = self.cell_id_queue[self.cell_id_queue != 0]
        cellblock_cache = [[None for x in range(len(self.merge_cellblock[0]))]
                            for y in range(len(self.merge_cellblock))] # for optimization. memorize local best merges

        while len(self.cell_id_queue) > 0:
            best_merge = {"next_cblock":np.copy(self.merge_cellblock),
                            "id":-1, "transform":0, "score":-np.inf, "pos":(-1,-1)}

            iteration_time = time.time()
            # for all possible positions
            for i in range(len(self.merge_cellblock)):
                for j in range(len(self.merge_cellblock)):
                    # check if cell can be activated
                    if (self.merge_cellblock[i][j][0] == -1
                        and self.adjacent_active_cell(self.merge_cellblock, i, j) != None):
                        cellblock_temp = np.copy(self.merge_cellblock)
                        cellblock_temp[i][j] = [1, 1]

                         # validate cellblock shape
                        if (self.validate_cellblock(cellblock_temp)):
                            # load from cache if possible
                            if (cellblock_cache[i][j] != None):
                                if (best_merge["score"]<cellblock_cache[i][j]["score"]):
                                    best_merge = cellblock_cache[i][j]
                                    cellblock_temp[i][j] = [cellblock_cache[i][j]["id"],
                                                            cellblock_cache[i][j]["transform"]]
                                    best_merge["next_cblock"] = cellblock_temp
                                continue

                            # if cache is not found
                            local_best_merge = self.best_fit_image_at(i, j, cellblock_temp)
                            if (best_merge["score"]<local_best_merge["score"]):
                                best_merge = local_best_merge
                            cellblock_cache[i][j] = local_best_merge

            if not np.any(self.cell_id_queue == best_merge["id"]): # safety measure
                print("FATAL ERROR: attempted to merge image id=",best_merge["id"], "terminating...")
                return
            # paste image to cellblock and update cache
            self.cell_id_queue = self.cell_id_queue[self.cell_id_queue != best_merge["id"]]
            self.merge_cellblock = best_merge["next_cblock"]
            self.merge_history.append(best_merge)
            self.clean_cellblock_cache(cellblock_cache, best_merge["pos"][0], best_merge["pos"][1], best_merge["id"])
            print("image merged: id =", best_merge["id"],
                    "t =", best_merge["transform"], "score=", np.round(best_merge["score"],4),"\t",
                    self.rows*self.cols-len(self.cell_id_queue), "/", self.rows*self.cols)
            #print(time.time() - iteration_time,"secs")
            #print("current-cellblock:\n", best_merge["next_cblock"][:,:,0])

        print("merge algorithm:",time.time()-s_time,"seconds")
