"""assembler_data_structures.py
helper data structures for assembler.py

CellData: representation of image cell including id, orientation and x,y position
CellBlock: blueprint for image assembly.
LHashmapPriorityQueue: linked-hashmap implementation of MST priority queue.
"""

import copy
import numpy as np

# direction ENUM (id, y delta, x delta)
DIR_ENUM = {
    'd': (0, 1, 0),
    'u': (1, -1, 0),
    'r': (2, 0, 1),
    'l': (3, 0, -1),
}


class CellData:
    """
    Representation of image cell (fragment). id, orientation state, position

    Attributes:
        img_id (int): image cell id
        transform (int): orientation state (flip, rotate) [0~3] for square cells,
                         [0~7] for rectangular ones.
        score (float): similarity score of which cell was stitched
        x (int): cell's x position inside the CellBlock
        y (int): cell's y position inside the CellBlock
        dir (int): stitching direction [0~3]

    Methods:
        deepcopy(CellData) -> CellData
        __str__() -> str
        __repr__() -> str
        __lt__(other: CellData) -> bool: overrides compare operator
        tostring() -> str
        set(id: int, transform: int, score: float, y: int, x: int, dir: int)
        is_valid() -> bool
        pos() -> y, x

    """

    def __init__(self, img_id=-1, transform=-1, score=-float("inf"), y=-1, x=-1, direction=-1):
        self.img_id = img_id  # unique id
        # transform: 0~3 for rectangular images (x_flip*y_flip),
        # 0~7 for square images (rotation90*x_flip*y_flip)
        self.transform = transform
        self.score = score  # similarity score at which cell was stitched
        self.y, self.x = y, x  # cell's position inside the cellblock
        self.dir = direction  # stitching direction, 0~3 ('d','u','r','l')

    def __str__(self):
        return str(self.img_id) if self.is_valid() else "-"

    def __repr__(self):
        return self.__str__()

    def tostring(self):
        """
        represent full instance state in string.
        """
        return ("{id: " + str(self.img_id) + ", t: " + str(self.transform) +
                ", (" + str(self.y) + ", " + str(self.x) + "), dir:" + str(self.dir) +
                ", score: " + format(self.score, ".4f") + " }")

    def set(self, img_id=-1, transform=-1, score=-float("inf"), y=-1, x=-1, direction=-1):
        """
        set selected attributes.
        """
        if img_id != -1:
            self.img_id = img_id
        if transform != -1:
            self.transform = transform
        if score > -float("inf"):
            self.score = score
        if y != -1 and x != -1:
            self.y, self.x = y, x
        if direction != -1:
            self.dir = direction
        return self

    def is_valid(self):
        """
        returns True if this Cell represents image,
        False if it only acts as a placeholder.
        (to be used within CellBlock)
        """
        return self.img_id >= 0

    def pos(self):
        """
        position of the cell within CellBlock
        """
        return self.y, self.x

    def __lt__(self, other):
        return self.score < other.score


class CellBlock:
    """
    Blueprint for image assembly.
    Holds 2d array of CellData, all initialized with dummy image id (-1)

    position is "inactive" if CellData has invalid id (<0)
    a position can be "activated" by inserting valid CellData (via CellBlock.activate_cell())

    Attributes:
        max_h: max allowed number of rows in assembled image
        max_w: max allowed number of columns in assembled image
        data: 2d list of CellData

    Methods:
        deepcopy(CellData) -> CellData
        __str__() -> str
        __repr__() -> str
        __lt__(other: CellData) -> bool: overrides comparator operator
        tostring() -> str
        set(id: int, transform: int, score: float, y: int, x: int, dir: int)
        is_valid() -> bool
        pos() -> y, x

    """

    def __init__(self, max_h, max_w):
        self._init = True
        self.max_h, self.max_w = max_h, max_w  # max allowed height and width
        w_h_size = max(max_h, max_w) * 2  # allocated width & height of data (2d list of CellData)
        # 2D array of celldata, blueprint for image reconstruction
        self.data = np.array([[CellData(y=i, x=j) for j in range(w_h_size)]
                              for i in range(w_h_size)])
        self.bottom = self.top = self.left = self.right = max(max_h, max_w)

    def active_neighbors(self, y, x):
        """
        returns all neighboring active Cells
        """
        adj = []
        for dirc in DIR_ENUM.values():
            if (0 < y + dirc[1] < len(self.data) and 0 < x + dirc[2] < len(self.data)
                    and self.data[y + dirc[1]][x + dirc[2]].is_valid()):
                adj.append(copy.deepcopy(self.data[y + dirc[1]][x + dirc[2]]).set(direction=dirc[0]))
        return adj

    def inactive_neighbors(self, y, x):
        """
        returns all neighboring inactive Cells
        """
        adj = []
        for dirc in DIR_ENUM.values():
            if (0 < y + dirc[1] < len(self.data) and 0 < x + dirc[2] < len(self.data)
                    and not self.data[y + dirc[1]][x + dirc[2]].is_valid()):
                if self.validate_pos(y + dirc[1], x + dirc[2]):
                    adj.append(self.data[y + dirc[1]][x + dirc[2]])
        return adj

    def validate_pos(self, y, x):
        """
        check if cell can be activated at position y, x
        """
        if (not self.data[y][x].is_valid() and self.active_neighbors(y, x)) or self._init:
            updated_h, updated_w = self.top - self.bottom, self.right - self.left
            if y < self.bottom or self.top < y:
                updated_h += 1
            if x < self.left or self.right < x:
                updated_w += 1
            return ((updated_h < self.max_h and updated_w < self.max_w) or
                    (updated_h < self.max_w and updated_w < self.max_h))
        return False

    def activate_cell(self, celldata):
        """
        activate cell at y, x
        NOTE: initial cell MUST be activated at y, x = max(max_h, max_w)

        @Parameters
        celldata (list):    [id, transform, score]
        """

        self._init = False
        self.data[celldata.y][celldata.x] = celldata
        if celldata.y > self.top:
            self.top += 1
        if celldata.y < self.bottom:
            self.bottom -= 1
        if celldata.x > self.right:
            self.right += 1
        if celldata.x < self.left:
            self.left -= 1

    def block_size(self):
        """
        returns current cellblock size of activated cells.
        """
        return self.top - self.bottom + 1, self.right - self.left + 1


class LHashmapPriorityQueue:
    """
    Linked Hashmap implementation for Priority Queue
    {key(int): val(Comparable object)}

                        max-heap vs linked-hashmap
    enqueue()           O(logN)     O(N)
    dequeue()           O(logN)     O(1)
    searchbyKey()       O(N)        O(1)

    Attributes:
        hashmap (dict)
        ll_head (Node)

    Methods:
        is_empty() -> bool
        peek() -> comparable object
        enqueue(key, val) -> void
        dequeue() -> comparable object
        dequeue_and_remove_duplicate_ids() -> comparable object, list of comparable objects

    """

    class Node:
        """
        doubly linked list node with key-val pair

        Attributes:
            key (str)
            val (object)
            next (Node)
            prev (Node)
        """

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
        self.hashmap = {i: [] for i in range(max_size)}
        self.ll_head = None  # linked list head

    def is_empty(self):
        """
        Returns:
            empty (bool)
        """
        return self.ll_head is None

    def peek(self):
        """
        Returns:
            val (Object)
        """
        return self.ll_head.val

    def enqueue(self, key, val):
        """
        Args:
            key (int):
            val (Object):
        """
        node = self.Node(key, val)
        self.hashmap[node.key].append(node)

        if self.ll_head is None:
            self.ll_head = node
            return
        if node.val > self.ll_head.val:
            node.next = self.ll_head
            self.ll_head = node
            node.next.prev = node
            return

        ptr = self.ll_head
        while ptr.next is not None:
            if node.val > ptr.next.val:
                node.next = ptr.next
                node.prev = ptr
                ptr.next = node
                node.next.prev = node
                return
            ptr = ptr.next
        ptr.next = node
        node.prev = ptr

    def dequeue(self):
        """
        Returns:
            node_val (object)
        """
        if not self.ll_head:
            return None
        return_node = self.ll_head

        self.ll_head = self.ll_head.next
        if return_node.next:
            return_node.next.prev = None
        self.hashmap[return_node.key].remove(return_node)
        return return_node.val

    def dequeue_and_remove_duplicate_ids(self):
        """
            dequeue top priority node & remove all nodes with the same key from the l_hashmap.
            runtime = O(N)
        """
        if not self.ll_head:
            return None
        return_node = self.ll_head

        duplicates_list = []
        for duplicate in self.hashmap[return_node.key]:
            duplicates_list.append(duplicate.val)
            if duplicate == self.ll_head:
                self.ll_head = self.ll_head.next
                if duplicate.next:
                    duplicate.next.prev = None
            else:
                duplicate.prev.next = duplicate.next
                if duplicate.next:
                    duplicate.next.prev = duplicate.prev
        self.hashmap[return_node.key] = []
        return return_node.val, duplicates_list
