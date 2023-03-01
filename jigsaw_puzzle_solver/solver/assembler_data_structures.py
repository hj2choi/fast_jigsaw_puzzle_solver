"""assembler_data_structures.py
data structures for assembler.py

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


class PuzzlePiece:
    """
    Representation of each puzzle pieces. holds id, orientation state, position information.

    Attributes:
        img_id (int): Unique id of the puzzle piece
        orientation (int): Orientation state of the puzzle piece (0-3 for square pieces, 0-7 for rectangular pieces)
        score (float): Similarity score of which this piece was stitched to the reconstructed image
        x (int): x position of the puzzle piece inside the construction blueprint
        y (int): y position of the puzzle piece inside the construction blueprint
        direction (int): Stitching direction (0-3 for 'd', 'u', 'r', 'l')

    Methods:
        deepcopy(PuzzlePiece) -> PuzzlePiece: Returns a deepcopy of the puzzle piece
        __lt__(other: PuzzlePiece) -> bool: Overrides the less-than operator to compare similarity score
        tostring() -> str: Returns a string representation of the puzzle piece
        set(id: int, orientation: int, score: float, y: int, x: int, direction: int)
        is_valid() -> bool: Returns True if the puzzle piece represents an image, False if it only acts as a placeholder
        pos() -> Tuple[int, int]: Returns the position of the puzzle piece inside the construction blueprint
    """

    def __init__(self, img_id=-1, orientation=-1, score=-float("inf"), y=-1, x=-1, direction=-1):
        self.img_id = img_id
        self.orientation = orientation
        self.score = score
        self.y, self.x = y, x
        self.dir = direction

    def __str__(self):
        return str(self.img_id) if self.is_valid() else "-"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, PuzzlePiece):
            return NotImplemented
        return self.img_id == other.img_id and self.orientation == other.orientation and \
            self.x == other.x and self.y == other.y and self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def tostring(self):
        """
        Returns a string representation of the puzzle piece
        """
        return ("{id: " + str(self.img_id) + ", t: " + str(self.orientation) +
                ", pos: (" + str(self.y) + ", " + str(self.x) + "), dir:" + str(self.dir) +
                ", score: " + format(self.score, ".4f") + " }")

    def set(self, img_id=-1, orientation=-1, score=-float("inf"), y=-1, x=-1, direction=-1):
        """
        Sets the attributes of the puzzle piece
        """
        if img_id != -1:
            self.img_id = img_id
        if orientation != -1:
            self.orientation = orientation
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
        """
        return self.img_id >= 0

    def pos(self):
        """
        position of the cell within CellBlock
        """
        return self.y, self.x


class PuzzleBlock:
    """
    Blueprint for assembling a jigsaw puzzle image.
    Holds 2d array of PuzzlePiece objects, each initialized with a blank image piece.

    - Positions in the array are considered "inactive"
      until they are activated by pasting a valid PuzzlePiece object via the activate_position() method.
    - A position can only be activated if it is adjacent to another activated position,
      and if the resulting assembly of activated positions does not exceed
      the maximum allowed size specified by max_h and max_w.

    Attributes:
        max_h (int): maximum allowed number of rows in the assembled image
        max_w (int): maximum allowed number of columns in the assembled image
        data (2d list of PuzzlePiece objects): the 2D array holding the PuzzlePiece objects
        bottom, top, left, right (int): the current boundaries of the activated positions in the array

    Methods:
        get_active_neighbors(y: int, x: int) -> list of PuzzlePiece: returns all neighboring active PuzzlePiece objects
        get_inactive_neighbors(y: int, x: int) -> list of PuzzlePiece: returns all neighboring inactive PuzzlePiece objects
        validate_position(y: int, x: int) -> bool: checks if position y, x can be activated
        activate_position(piece: PuzzlePiece): activates position y, x by pasting PuzzlePiece object at y, x
        block_size() -> height, width: returns the current blueprint size of activated positions
    """

    def __init__(self, max_h, max_w):
        self._init = True  # flag indicating whether this is the first PuzzlePiece to be added
        self.max_h, self.max_w = max_h, max_w  # maximum allowed height and width
        w_h_size = max(max_h, max_w) * 2  # allocated width and height of data array
        # allocate 2D array of PuzzlePiece objects
        self.data = np.array([[PuzzlePiece(y=i, x=j) for j in range(w_h_size)]
                              for i in range(w_h_size)])
        self.bottom = self.top = self.left = self.right = w_h_size // 2  # first position

    def get(self, y, x):
        return self.data[y][x]

    def get_active_neighbors(self, y, x):
        """
        Returns a list of neighboring active PuzzlePiece objects.

        Args:
            y (int): the y-coordinate of the position
            x (int): the x-coordinate of the position

        Returns:
            List[PuzzlePiece]: a list of neighboring active PuzzlePiece objects
        """
        adj = []
        for dirc in DIR_ENUM.values():
            if (0 < y + dirc[1] < len(self.data) and 0 < x + dirc[2] < len(self.data)
                    and self.data[y + dirc[1]][x + dirc[2]].is_valid()):
                adj.append(copy.deepcopy(self.data[y + dirc[1]][x + dirc[2]]).set(direction=dirc[0]))
        return adj

    def get_inactive_neighbors(self, y, x):
        """
        Returns a list of neighboring inactive PuzzlePiece objects.
        """
        adj = []
        for dirc in DIR_ENUM.values():
            if (0 < y + dirc[1] < len(self.data) and 0 < x + dirc[2] < len(self.data)
                    and not self.data[y + dirc[1]][x + dirc[2]].is_valid()):
                if self.validate_position(y + dirc[1], x + dirc[2]):
                    adj.append(self.data[y + dirc[1]][x + dirc[2]])
        return adj

    def validate_position(self, y, x):
        """
        check if the given position y, x can be activated (i.e., a puzzle piece can be placed there).
        A position is valid if it is not already occupied and has at least one adjacent activated position.
        """
        if (not self.data[y][x].is_valid() and self.get_active_neighbors(y, x)) or self._init:
            # Calculate the updated block size if this position is activated.
            updated_h, updated_w = self.top - self.bottom, self.right - self.left
            if y < self.bottom or self.top < y:
                updated_h += 1
            if x < self.left or self.right < x:
                updated_w += 1
            # Check if the updated block size is within the maximum dimensions.
            return ((updated_h < self.max_h and updated_w < self.max_w) or
                    (updated_h < self.max_w and updated_w < self.max_h))
        return False

    def activate_position(self, piece):
        """
        Activate the given puzzle piece by placing it in the corresponding position in the block.
        NOTE: the first piece MUST be pasted at y, x = w_h_size/2, w_h_size/2

        @Parameters
        piece (PuzzlePiece)
        """

        self._init = False
        self.data[piece.y][piece.x] = piece

        # Update the block boundaries based on the position of the newly activated piece.
        if piece.y > self.top:
            self.top += 1
        if piece.y < self.bottom:
            self.bottom -= 1
        if piece.x > self.right:
            self.right += 1
        if piece.x < self.left:
            self.left -= 1

    def block_size(self):
        """
        Returns the current dimensions of the block that contains activated positions.
        """
        return self.top - self.bottom + 1, self.right - self.left + 1


class LinkedHashmapPriorityQueue:
    """
    Linked Hashmap implementation of Priority Queue
    {key(int): val(Comparable object)}

                        max-heap vs linked-hashmap
    enqueue()           O(logN)     O(N)
    dequeue()           O(logN)     O(1)
    search_by_key()     O(N)        O(1)

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
