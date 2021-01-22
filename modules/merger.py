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

class Cellblock:
    def __init__(self, max_h, max_w, cells_cnt, data = None, cell_pos = None, hs = -1, ht = -1, ws = -1, wt = -1, init = True):
        self.max_h, self.max_w = max_h, max_w # max allowed h,w
        self.cells_cnt = cells_cnt # number of loaded images
        self.length = max(max_h, max_w)*2
        self.data = data # blueprint for image reconstruction
        self.cell_pos = cell_pos # position of each cell IDs
        self.hs, self.ht = hs, ht # h start - terminal
        self.ws, self.wt = ws, wt # w start - terminal
        if init:
            self.data = np.array([[[-1, -1, -1.] for j in range(self.length)]
                                                for i in range(self.length)],
                                                dtype = object)
            half_length = max(max_h, max_w)
            self.data[half_length][half_length].fill(int(0))
            self.hs, self.ht = half_length, half_length # h start - terminal
            self.ws, self.wt = half_length, half_length # w start - terminal
            self.cell_pos = {i:[] for i in range(self.cells_cnt)}
            self.cell_pos[0] = [(half_length, half_length)]

    @classmethod
    def copy(cls, cellblock):
        return cls(cellblock.max_h, cellblock.max_w, cellblock.cells_cnt,
                     np.copy(cellblock.data), cellblock.cell_pos.copy(),
                    cellblock.hs, cellblock.ht, cellblock.ws, cellblock.wt,
                    False)

    """
    finds all adjacent active cells. (active cell: cell with id >= 0)
    @Returns
    adj_dirs (npArray):     list of all directions to adjacent active cell [d, u, l, r]
    """
    def adjacent_active_cells(self, i, j):
        adj = []
        if (i+1 < len(self.data) and self.data[i+1][j][0] >= 0):
            adj.append({"dir":DIR['d'], "id":self.data[i+1][j][0], "transform":self.data[i+1][j][1]})
        if (i-1 > 0 and self.data[i-1][j][0] >= 0):
            adj.append({"dir":DIR['u'], "id":self.data[i-1][j][0], "transform":self.data[i-1][j][1]})
        if (j+1 < len(self.data[0]) and self.data[i][j+1][0] >= 0):
            adj.append({"dir":DIR['r'], "id":self.data[i][j+1][0], "transform":self.data[i][j+1][1]})
        if (j-1 > 0 and self.data[i][j-1][0] >= 0):
            adj.append({"dir":DIR['l'], "id":self.data[i][j-1][0], "transform":self.data[i][j-1][1]})
        return adj

    """
    finds single adjacent active cell.
    """
    def adjacent_to_active_cell(self, i, j):
        return ((i+1 < len(self.data) and self.data[i+1][j][0] >= 0) or
                (i-1 > 0 and self.data[i-1][j][0] >= 0) or
                (j+1 < len(self.data[0]) and self.data[i][j+1][0] >= 0) or
                (j-1 > 0 and self.data[i][j-1][0] >= 0))

    """
    validate if cell can be activated at position
    """
    def validate_cell_pos(self, y, x):
        if not (self.data[y][x][0] == -1 and self.adjacent_to_active_cell(y, x)):
            return False
        updated_h, updated_w = self.ht - self.hs, self.wt - self.ws
        if y < self.hs or self.ht < y: updated_h += 1
        if x < self.ws or self.wt < x: updated_w += 1
        return ((updated_h < self.max_h and updated_w < self.max_w) or
                (updated_h < self.max_w and updated_w < self.max_h))

    """
    activate cell at y, x

    @Parameters
    celldata (list):    [id, transform, score]
    """
    def activate_cell(self, celldata, y, x):
        self.data[y][x] = celldata
        self.cell_pos[celldata[0]].append((y, x))
        if y > self.ht: self.ht+=1
        if y < self.hs: self.hs-=1
        if x > self.wt: self.wt+=1
        if x < self.ws: self.ws-=1

    """
    cleanup and update cache
    """
    def clean_cache(self, y, x, removed_id):
        if x+1 < len(self.data[0]): self.data[y][x+1] = [-1,-1,-1]
        if y+1 < len(self.data): self.data[y+1][x] = [-1,-1,-1]
        if x-1 > 0: self.data[y][x-1] = [-1,-1,-1]
        if y-1 > 0: self.data[y-1][x] = [-1,-1,-1]
        if self.cell_pos[removed_id] != None:
            pos = self.cell_pos[removed_id]
            for pos in self.cell_pos[removed_id]:
                self.data[pos[0]][pos[1]] = [-1,-1,-1]
            self.cell_pos[removed_id] = []

    def size(self):
        return (self.ht - self.hs + 1, self.wt - self.ws + 1)



class ImageMerger:
    """
    ImageMerger constructor. To be used with loadfromfilepath()

    @usage
    mr = mr.ImageMerger.loadfromfilepath(args, ...)
    """
    def __init__(self, data = [], cols = 0, rows = 0):
        self.img_cells = data # image slices with all combination of transformations
        self.t_count = len(self.img_cells[0]) # number of possible transformation states
        self.cell_id_queue = np.arange(0,len(self.img_cells))# list of image ids to be processed
        self.merge_cellblock = (np.zeros((max(rows, cols)*2,max(rows, cols)*2,2))-1).astype(int)  # blueprint for image reconstruction
        self.cols = cols # as specified in args
        self.rows = rows # as specified in args
        if (self.t_count == 8):
            self.sim_matrix = np.zeros((len(self.img_cells), len(self.img_cells), 32))
            self.map = opt.map8 # similarity matrix index mapper (flip, mirror, rotation)
        else:
            self.sim_matrix = np.zeros((len(self.img_cells), len(self.img_cells), 16))
            self.map = opt.map4 # similarity matrix index mapper (flip, mirror)
        self.merge_history = [] # used for merge animation

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
                    if (len(img) == len(img[0])): # square images
                        all_transformations = []
                        for i in range(8):
                            if i == 4:
                                img = np.flip(img, 0)
                            all_transformations.append(np.copy(np.rot90(img, i)))
                        img_cells.append(all_transformations)
                    else: # rectangle images
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
        cellblock = self.merge_cellblock.data
        t_cnt = self.t_count
        rt, ct, rs, cs = self.merge_cellblock.ht, self.merge_cellblock.wt, self.merge_cellblock.hs, self.merge_cellblock.ws
        cell_h = len(self.img_cells[0][0])
        cell_w = len(self.img_cells[0][0][0])
        cellblock_h = (rt - rs + 1)*cell_h
        cellblock_w = (ct - cs + 1)*cell_w

        whiteboard = np.zeros((cellblock_h, cellblock_w, 3), dtype = np.uint8)
        whiteboard.fill(0)
        for i in range(len(cellblock)):
            for j in range(len(cellblock[0])):
                if (cellblock[i][j][0] > -1):
                    idx = cellblock[i][j]
                    paste = self.img_cells[idx[0]][idx[1] % t_cnt]
                    y_offset = (i-rs)*cell_h
                    x_offset = (j-cs)*cell_w
                    whiteboard[y_offset:y_offset+cell_h, x_offset:x_offset+cell_w] = paste

        cv2.imwrite(filepath+".png", whiteboard)

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
        # try running in parallel. Run as serial otherwise.
        if (self.construct_similaritymatrix_parallel(img_cells_norm, t_cnt)):
            return

        s_time = time.time()
        for i in range(len(self.img_cells)):
            for j in range(len(self.img_cells)):
                if (i == j):
                    continue
                for k in range(len(self.sim_matrix[i][j])):
                    self.sim_matrix[i][j][k] = im_op.img_borders_similarity(img_cells_norm[j][(k%epoch)%t_cnt],
                                            img_cells_norm[i][k//epoch], (k%epoch)//t_cnt)
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
    find best-fit image at given position
    """
    def best_fit_image_at(self, i, j):
        t_cnt = self.t_count
        local_best_img = {"cell": [-1, 0, -np.inf], "pos":(-1,-1)}
        for id in self.cell_id_queue: # for all images in queue
            for adj in self.merge_cellblock.adjacent_active_cells(i, j): # for all adjacent images
                for k in range(t_cnt*adj["dir"], t_cnt*adj["dir"]+t_cnt): # for all transformations
                    score = self.sim_matrix[adj["id"]][id][self.map(adj["transform"], k)]
                    if (local_best_img["cell"][0]==-1 or local_best_img["cell"][2]<score):
                        local_best_img = {"cell": [id, k%t_cnt, score], "pos": (i,j)}
        return local_best_img

    """
    merge image cells back to original image.

    1. place all images in a queue.
    2. start with a empty 'board' and paste random image at (r=0, c=0)
    2. while queue is not empty:
        2-1. for all possible position for any image to be pasted:
            2-1-1. find best-fit image, and its transformation and location
                    in accordance to similarity matrix
            2-1-2. save local best-fit image to cache (optimization)
        2-2. paste best-fit image to the 'board'
        2-3. remove image from queue
    3. construct final image

    @Requirements:
    sim_matrix (npArray):   similarity matrix for all image pairs.
                            similarity for all possible transformations and stitching directions should be pre-computed
    """
    def merge(self):
        self.construct_similaritymatrix()
        s_time = time.time()
        t_cnt = self.t_count
        self.cell_id_queue = self.cell_id_queue[self.cell_id_queue != 0]
        self.merge_cellblock = Cellblock(self.rows, self.cols, len(self.cell_id_queue)+1)
        cellblock_cache = Cellblock(self.rows, self.cols, len(self.cell_id_queue)+1)# for optimization. memorize local best merges
        while len(self.cell_id_queue) > 0:
            best_merge = {"next_cblock":Cellblock.copy(self.merge_cellblock),
                            "id":-1, "transform":0, "score":-np.inf, "pos":(-1,-1)}

            iteration_time = time.time()
            # for all possible positions
            for i in range(self.merge_cellblock.length):
                for j in range(self.merge_cellblock.length):
                    # if cell can be activated at position i, j
                    if (self.merge_cellblock.validate_cell_pos(i, j)):
                        cellblock_temp = Cellblock.copy(self.merge_cellblock)

                        # cache hit
                        if (cellblock_cache.data[i][j][0] != -1):
                            if (best_merge["score"]<cellblock_cache.data[i][j][2]):
                                cellblock_temp.activate_cell(cellblock_cache.data[i][j], i, j)
                                best_merge = {"next_cblock": cellblock_temp,
                                                "id": cellblock_cache.data[i][j][0],
                                                "transform": cellblock_cache.data[i][j][1],
                                                "score": cellblock_cache.data[i][j][2],
                                                "pos":(i, j)}
                                best_merge["next_cblock"] = cellblock_temp
                            continue

                        # cache miss
                        local_best_img = self.best_fit_image_at(i, j) # id, transform, score
                        cellblock_temp.activate_cell(local_best_img["cell"], i, j)
                        local_best_merge = {"next_cblock": cellblock_temp,
                                        "id": local_best_img["cell"][0],
                                        "transform": local_best_img["cell"][1],
                                        "score": local_best_img["cell"][2],
                                        "pos":local_best_img["pos"]}
                        if (best_merge["score"]<local_best_img["cell"][2]):
                            best_merge = local_best_merge
                        cellblock_cache.activate_cell([local_best_merge["id"], local_best_merge["transform"],
                                                        local_best_merge["score"]], i, j)

            if not np.any(self.cell_id_queue == best_merge["id"]): # fail-safe
                print("merge failed: attempted to merge image id=",best_merge["id"], "terminating...")
                return
            # merge best-fit image to cellblock and update cache
            self.cell_id_queue = self.cell_id_queue[self.cell_id_queue != best_merge["id"]]
            self.merge_cellblock = best_merge["next_cblock"]
            self.merge_history.append(best_merge)
            cellblock_cache.clean_cache(best_merge["pos"][0], best_merge["pos"][1], best_merge["id"])
            print("image merged: id =", best_merge["id"],
                    "t =", best_merge["transform"], "score=", np.round(best_merge["score"],4),"\t",
                    self.rows*self.cols-len(self.cell_id_queue), "/", self.rows*self.cols)
            #print(time.time() - iteration_time,"secs")
            #print("current-cellblock:\n", best_merge["next_cblock"].data[:,:,0])

        print("merge algorithm:",time.time()-s_time,"seconds")
