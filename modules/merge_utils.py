from . import img_operations as im_op
import numpy as np

#direction ENUM
DIR = {
    'd': (0, 1, 0),
    'u': (1, -1, 0),
    'r': (2, 0, 1),
    'l': (3, 0, -1),
}
#switch to parallel computation if data size reaches image count threshold
MAX_PROCESS_COUNT = 3
SWITCH_TO_PARALLEL_THRESHOLD = 128 # number of images

"""
Representation of image cell. id, orientation state, position
"""
class CellData:
    def __init__(self, id = -1, transform = -1, score = -float("inf"), y = -1, x = -1, dir = -1):
        self.id = id
        self.transform = transform
        self.score = score
        self.y, self.x = y, x
        self.dir = dir #stitching direction
    @classmethod
    def copy(cls, celldata):
        return cls(celldata.id, celldata.transform, celldata.score, celldata.y, celldata.x, celldata.dir)

    def __str__(self):
        return str(self.id) if self.is_valid() else "-"
    def __repr__(self):
        return self.__str__()
    def tostring(self):
        return ("{id: " + str(self.id) + ", t: " + str(self.transform) +
                ", (" + str(self.y) + ", " + str(self.x) + "), score: " + format(self.score, ".4f") + " }")

    def set(self, id = -1, transform = -1, score = -float("inf"), y = -1, x = -1, dir = -1):
        if id != -1: self.id = id
        if transform != -1: self.transform = transform
        if score > -float("inf"): self.score = score
        if y != -1 and x != -1: self.y, self.x = y, x
        if dir != -1: self.dir = dir
        return self

    def is_valid(self):
        return self.id >= 0

    def pos(self):
        return (self.y, self.x)

    def __lt__(self, other):
        return self.score < other.score



"""
blueprint for image reconstruction
"""
class CellBlock:
    def __init__(self, max_h, max_w, data = None, hs = -1, ht = -1, ws = -1, wt = -1, init = True):
        self.init = init # true if no cell has been activated yet
        self.max_h, self.max_w = max_h, max_w # max allowed h,w
        self.length = max(max_h, max_w)*2 # cellblock data width & height
        self.data = data # blueprint for image reconstruction
        self.hs, self.ht = hs, ht # h start - terminal
        self.ws, self.wt = ws, wt # w start - terminal
        if self.init:
            self.data = np.array([[CellData(y = i, x = j) for j in range(self.length)]
                                    for i in range(self.length)])
            half_length = max(max_h, max_w)
            self.hs = self.ht = self.ws = self.wt = half_length

    @classmethod
    def copy(cls, cblock):
        return cls(cblock.max_h, cblock.max_w, np.copy(cblock.data),
                    cblock.hs, cblock.ht, cblock.ws, cblock.wt, False)

    def active_neighbors(self, i, j):
        adj = []
        for d in DIR.values():
            if (0 < i + d[1] and i + d[1] < self.length and 0 < j + d[2] and j + d[2] < self.length
                and self.data[i + d[1]][j + d[2]].is_valid()):
                adj.append(self.data[i + d[1]][j + d[2]].set(dir = d[0]))
        return adj

    def inactive_neighbors(self, i, j):
        adj = []
        for d in DIR.values():
            if (0 < i + d[1] and i + d[1] < self.length and 0 < j + d[2] and j + d[2] < self.length
                and not self.data[i + d[1]][j + d[2]].is_valid()):
                if self.validate_pos(i + d[1], j + d[2]): adj.append(self.data[i + d[1]][j + d[2]])
        return adj

    """
    validate if cell can be activated at position y, x
    """
    def validate_pos(self, y, x):
        if (not self.data[y][x].is_valid() and 0 < len(self.active_neighbors(y, x))) or self.init:
            updated_h, updated_w = self.ht - self.hs, self.wt - self.ws
            if y < self.hs or self.ht < y: updated_h += 1
            if x < self.ws or self.wt < x: updated_w += 1
            return ((updated_h < self.max_h and updated_w < self.max_w) or
                    (updated_h < self.max_w and updated_w < self.max_h))
        return False

    """
    activate cell at y, x
    NOTE: initial cell MUST be activated at y, x = max(max_h, max_w)

    @Parameters
    celldata (list):    [id, transform, score]
    """
    def activate_cell(self, celldata):
        self.init = False
        self.data[celldata.y][celldata.x] = celldata
        if celldata.y > self.ht: self.ht += 1
        if celldata.y < self.hs: self.hs -= 1
        if celldata.x > self.wt: self.wt += 1
        if celldata.x < self.ws: self.ws -= 1

    def size(self):
        return (self.ht - self.hs + 1, self.wt - self.ws + 1)


'''
Linked Hashmap implementation for Priority Queue
{key(int): val(Comparable)}

                    max-heap vs linked-hashmap
enqueue()           O(logN)     O(N)
dequeue()           O(logN)     O(1)
searchbyKey()       O(N)        O(1)
'''
class LHashmapPriorityQueue:
    class Node:
        def __init__(self, key, val):
            self.key = key
            self.val = val
            self.next = None
            self.prev = None
        def __str__(self):
            return str(self.val)
        def __repr__(self):
            return self.__str__()

    def __init__(self, max_size):
        self.hashmap = {i:[] for i in range(max_size)}
        self.ll_head = None # linked list head

    def is_empty(self):
        return self.ll_head == None

    def peek(self):
        return self.ll_head.val

    def enqueue(self, key, val):
        node = self.Node(key, val)
        self.hashmap[node.key].append(node)

        if self.ll_head == None:
            self.ll_head = node
            return
        if node.val > self.ll_head.val:
            node.next = self.ll_head
            self.ll_head = node
            node.next.prev = node
            return

        p = self.ll_head
        while p.next != None:
            if node.val > p.next.val:
                node.next = p.next
                node.prev = p
                p.next = node
                node.next.prev = node
                return
            p = p.next
        p.next = node
        node.prev = p

    def dequeue(self):
        if not self.ll_head:
            return None
        returnnode = self.ll_head

        self.ll_head = self.ll_head.next
        if returnnode.next:
            returnnode.next.prev = None
        self.hashmap[returnnode.key].remove(returnnode)
        return returnnode.val

    def dequeue_and_remove_duplicate_ids(self):
        if not self.ll_head:
            return None
        returnnode = self.ll_head

        duplicates_list = []
        for duplicate in self.hashmap[returnnode.key]:
            duplicates_list.append(duplicate.val)
            if duplicate == self.ll_head:
                self.ll_head = self.ll_head.next
                if duplicate.next:
                    duplicate.next.prev = None
            else:
                duplicate.prev.next = duplicate.next
                if duplicate.next:
                    duplicate.next.prev = duplicate.prev
        self.hashmap[returnnode.key] = []
        return returnnode.val, duplicates_list

"""
given a fixed orientation of tgt,
possible src orientations = rotation x flip x 4directions = 4 * 2 * 4 = 32
             [0~7]
              src
  [16~23] src tgt src [24~31]
              src
             [8~15]
total possible cases = 32 x 8 tgt orientations = 32 x 8 = 256
below is the mapping table for [256 possible cases] => [32 cases with fixed tgt orientation]
"""
_MAPPING_TABLE8 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                27, 24, 25, 26, 31, 28, 29, 30, 19, 16, 17, 18, 23, 20, 21, 22, 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14,
                10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 26, 27, 24, 25, 30, 31, 28, 29, 18, 19, 16, 17, 22, 23, 20, 21,
                17, 18, 19, 16, 21, 22, 23, 20, 25, 26, 27, 24, 29, 30, 31, 28, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4,
                12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1, 20, 23, 22, 21, 16, 19, 18, 17, 28, 31, 30, 29, 24, 27, 26, 25,
                29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
                6, 5, 4, 7, 2, 1, 0, 3, 14, 13, 12, 15, 10, 9, 8, 11, 30, 29, 28, 31, 26, 25, 24, 27, 22, 21, 20, 23, 18, 17, 16, 19,
                23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]
"""
mapping for [64 cases] => [16 cases with fixed tgt]
"""
def _flip_x(i):
    return [1,0,3,2,5,4,7,6,13,12,15,14,9,8,11,10][i]
def _flip_y(i):
    return [6,7,4,5,2,3,0,1,10,11,8,9,14,15,12,13][i]

def map8(tgt_transform, src_transform):
    return _MAPPING_TABLE8[tgt_transform * 32 + src_transform]

def map4(tgt_transform, src_transform):
    return {
        0:src_transform, 1:_flip_x(src_transform), 2:_flip_y(src_transform), 3:_flip_x(_flip_y(src_transform))
    }[tgt_transform]


"""
Test routines
"""
def _testpq_linkedlist(llhashmap):
    print("linked-list test:")
    p = llhashmap.ll_head
    while p != None:
        print(p)
        if (p.next and p.next.prev != p):
            print("ERROR: wrong pointer to .prev", p.next.prev)
        p = p.next

def _testpq():
    print("test llhashmap started")
    llhashmap = LHashmapPriorityQueue(10)
    llhashmap.enqueue(0, 3.2) # key, val
    llhashmap.enqueue(1, 4.2)
    llhashmap.enqueue(2, 3.5)
    llhashmap.enqueue(1, 2.2)
    llhashmap.enqueue(3, 3)
    llhashmap.enqueue(3, 1)
    llhashmap.enqueue(2, 5)
    llhashmap.enqueue(0, 3.3)
    print(llhashmap.hashmap)
    llhashmap._testll()
    print("pop")
    llhashmap.dequeue()
    llhashmap._testll()
    print("pop")
    llhashmap.dequeue()
    llhashmap._testll()
    print("pop")
    llhashmap.dequeue()
    llhashmap._testll()
    print("pop")
    llhashmap.dequeue()
    print(llhashmap.hashmap)
    llhashmap._testll()

def _generate_mapping_table8(sim_matrix):
    mapping_table = []
    for i in range(len(sim_matrix[0][1])):
        flag = 0
        for j in range(32):
            print(i,j,"=>",sim_matrix[0][2][i],sim_matrix[0][2][j], end="")
            if (np.abs(sim_matrix[0][2][i] - sim_matrix[0][2][j]) < 0.0000001):
                if not flag:
                    print(" match", end="")
                    flag = 1
                    mapping_table.append(j)
                else:
                    print("\n SOMETHING's WRONG")
                    return
            print()
        print()

    print(mapping_table)
