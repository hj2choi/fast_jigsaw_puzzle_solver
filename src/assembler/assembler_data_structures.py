import numpy as np

# direction ENUM
DIR = {
    'd': (0, 1, 0),
    'u': (1, -1, 0),
    'r': (2, 0, 1),
    'l': (3, 0, -1),
}


class CellData:
    """
    Representation of image cell (fragment). id, orientation state, position

    Attributes:
        id (int): image cell id
        transform (int): orientation state (flip, rotate) [0~3] for square cells, [0~7] for rectangular ones.
        score (float): similarity score of which cell was stitched
        x (int): cell's x position inside the CellBlock
        y (int): cell's y position inside the CellBlock
        dir (int): stitching direction [0~3]

    Methods:
        copy(CellData) -> CellData
        __str__() -> str
        __repr__() -> str
        __lt__(other: CellData) -> bool: overrides compar operator
        tostring() -> str
        set(id: int, transform: int, score: float, y: int, x: int, dir: int)
        is_valid() -> bool
        pos() -> y, x

    """

    def __init__(self, id=-1, transform=-1, score=-float("inf"), y=-1, x=-1, dir=-1):
        self.id = id  # unique id
        # transform: 0~3 for rectangular images (x_flip*y_flip), 0~7 for square images (rotation90*x_flip*y_flip)
        self.transform = transform
        self.score = score  # similarity score at which cell was stitched
        self.y, self.x = y, x  # cell's position inside the cellblock
        self.dir = dir  # stitching direction, 0~3 ('d','u','r','l')

    @classmethod
    def copy(cls, celldata):
        # deep copy
        return cls(celldata.id, celldata.transform, celldata.score, celldata.y, celldata.x, celldata.dir)

    def __str__(self):
        return str(self.id) if self.is_valid() else "-"

    def __repr__(self):
        return self.__str__()

    def tostring(self):
        return ("{id: " + str(self.id) + ", t: " + str(self.transform) +
                ", (" + str(self.y) + ", " + str(self.x) + "), dir:" + str(self.dir) +
                ", score: " + format(self.score, ".4f") + " }")

    def set(self, id=-1, transform=-1, score=-float("inf"), y=-1, x=-1, dir=-1):
        # set selected attributes.
        if id != -1:
            self.id = id
        if transform != -1:
            self.transform = transform
        if score > -float("inf"):
            self.score = score
        if y != -1 and x != -1:
            self.y, self.x = y, x
        if dir != -1:
            self.dir = dir
        return self

    def is_valid(self):
        return self.id >= 0

    def pos(self):
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
        copy(CellData) -> CellData
        __str__() -> str
        __repr__() -> str
        __lt__(other: CellData) -> bool: overrides comparator operator
        tostring() -> str
        set(id: int, transform: int, score: float, y: int, x: int, dir: int)
        is_valid() -> bool
        pos() -> y, x

    """

    def __init__(self, max_h, max_w, data=None, hs=-1, ht=-1, ws=-1, wt=-1, init=True):
        self._init = init  # true if no cell has been activated yet
        self.max_h, self.max_w = max_h, max_w  # max allowed height and width
        self._length = max(max_h, max_w) * 2  # cellblock 2D array width & height
        self.data = data  # 2D array of celldata, blueprint for image reconstruction
        self.hs, self.ht = hs, ht  # h start - end
        self.ws, self.wt = ws, wt  # w start - end
        if self._init:
            self.data = np.array([[CellData(y=i, x=j) for j in range(self._length)] for i in range(self._length)])
            half_length = max(max_h, max_w)
            self.hs = self.ht = self.ws = self.wt = half_length

    @classmethod
    def copy(cls, cblock):
        return cls(cblock.max_h, cblock.max_w, np.copy(cblock.data),
                   cblock.hs, cblock.ht, cblock.ws, cblock.wt, False)

    def active_neighbors(self, y, x):
        adj = []
        for d in DIR.values():
            if (0 < y + d[1] < self._length and 0 < x + d[2] < self._length
                    and self.data[y + d[1]][x + d[2]].is_valid()):
                adj.append(CellData.copy(self.data[y + d[1]][x + d[2]]).set(dir=d[0]))
        return adj

    def inactive_neighbors(self, y, x):
        adj = []
        for d in DIR.values():
            if (0 < y + d[1] < self._length and 0 < x + d[2] < self._length
                    and not self.data[y + d[1]][x + d[2]].is_valid()):
                if self.validate_pos(y + d[1], x + d[2]): adj.append(self.data[y + d[1]][x + d[2]])
        return adj

    def validate_pos(self, y, x):
        """
            check if cell can be activated at position y, x
        """
        if (not self.data[y][x].is_valid() and 0 < len(self.active_neighbors(y, x))) or self._init:
            updated_h, updated_w = self.ht - self.hs, self.wt - self.ws
            if y < self.hs or self.ht < y:
                updated_h += 1
            if x < self.ws or self.wt < x:
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
        if celldata.y > self.ht:
            self.ht += 1
        if celldata.y < self.hs:
            self.hs -= 1
        if celldata.x > self.wt:
            self.wt += 1
        if celldata.x < self.ws:
            self.ws -= 1

    def size(self):
        return self.ht - self.hs + 1, self.wt - self.ws + 1


class LHashmapPriorityQueue:
    """
    Linked Hashmap implementation for Priority Queue
    {key(int): val(Comparable)}

                        max-heap vs linked-hashmap
    enqueue()           O(logN)     O(N)
    dequeue()           O(logN)     O(1)
    searchbyKey()       O(N)        O(1)

    Attributes:
        hashmap (dict)
        ll_head (Node)

    Methods:
        is_empty() -> bool
        peek() -> object
        enqueue(key, val) -> void
        dequeue() -> object
        dequeue_and_remove_duplicate_ids() -> object, list of objects

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
        return self.ll_head is None

    def peek(self):
        return self.ll_head.val

    def enqueue(self, key, val):
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

        p = self.ll_head
        while p.next is not None:
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
