import unittest
from jigsaw_puzzle_solver.assembler.assembler_data_structures import CellData, CellBlock, LHashmapPriorityQueue


class CellDataTest(unittest.TestCase):
    def setUp(self):
        self.cell = CellData()

    def tearDown(self):
        return

    def test_constructor(self):
        result = [self.cell.img_id,
                  self.cell.transform,
                  self.cell.score,
                  self.cell.y,
                  self.cell.x,
                  self.cell.dir]
        expected = [-1, -1, -float("inf"), -1, -1, -1]
        self.assertEqual(result, expected, "incorrect default initialization")

        self.cell = CellData(1, 2, 3, 4, 5, 6)
        result = [self.cell.img_id,
                  self.cell.transform,
                  self.cell.score,
                  self.cell.y,
                  self.cell.x,
                  self.cell.dir]
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected, "incorrect initialization")

    def test_set(self):
        self.cell = CellData(1, 2, 3, 4, 5, 6)
        self.cell.set() # do not change anything
        result = [self.cell.img_id,
                  self.cell.transform,
                  self.cell.score,
                  self.cell.y,
                  self.cell.x,
                  self.cell.dir]
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(result, expected,
                         "CellData.set() without parameters should not change anything")

        self.cell = CellData(1, 2, 3, 4, 5, 6)
        self.cell.set(transform=7, score=0.9, direction=2)  # partially set values
        result = [self.cell.img_id,
                  self.cell.transform,
                  self.cell.score,
                  self.cell.y,
                  self.cell.x,
                  self.cell.dir]
        expected = [1, 7, 0.9, 4, 5, 2]
        self.assertEqual(result, expected, "incorrect CellData.set() behavior")

        self.cell = CellData(1, 2, 3, 4, 5, 6)
        self.cell.set(11, 12, 13, 14, 15, 16)  # set all values
        result = [self.cell.img_id,
                  self.cell.transform,
                  self.cell.score,
                  self.cell.y,
                  self.cell.x,
                  self.cell.dir]
        expected = [11, 12, 13, 14, 15, 16]
        self.assertEqual(result, expected, "incorrect CellData.set() behavior")

    def test_is_valid(self):
        # test if cell has id > 0
        cell_1 = CellData()
        cell_2 = CellData(img_id=0)
        cell_3 = CellData(img_id=132412343)
        result = [cell_1.is_valid(),
                  cell_2.is_valid(),
                  cell_3.is_valid()]
        expected = [False, True, True]
        self.assertEqual(result, expected, "incorrect CellData.is_valid() behavior")

    def test_get_pos(self):
        cell = CellData(y=10, x=11)
        result = [cell.y,
                  cell.x]
        expected = [10, 11]
        self.assertEqual(result, expected, "incorrect CellData.pos() behavior")

    def test_less_than_comparator(self):
        # test overridden comparator operator
        cell1 = CellData(score=0.8)
        cell2 = CellData(score=0.9)
        result = [cell1 < cell2,
                  cell1 > cell2]
        expected = [True, False]
        self.assertEqual(result, expected, "incorrect >, < comparator behavior")


class CellBlockTest(unittest.TestCase):
    def setUp(self):
        self.cellblock = CellBlock(3, 4)

    def tearDown(self):
        return

    def test_constructor(self):
        result = [self.cellblock.max_h, self.cellblock.max_w,
                  self.cellblock.bottom, self.cellblock.top, self.cellblock.left, self.cellblock.right
                  ]
        expected = [3, 4, 4, 4, 4, 4]
        self.assertEqual(result, expected, "incorrect constructor behavior")

    def test_activate_cell(self):
        cblock = CellBlock(2, 4)
        cell = CellData()