# Fast Jigsaw Puzzle Solver with Unknown Orientation

- Splits an image into **N** (rows x columns) anonymized rectangular pieces with random orientations.
- Reconstructs the **N** puzzle pieces back to original image in **O(N²)** time complexity.

<img src="https://github.com/user-attachments/assets/89dd6583-fddf-4c1b-8cac-b87953466b3b" alt="jigsaw_puzzle_solver_demo" width="90%">

*Disclaimer: The orientation of the reconstructed image may be random. Successful reconstruction is not always guaranteed.*

## Features

- Uses the Euclidean distance metric for matching image boundaries.
- Computes 3D distance matrix in **Parallel** (a rank-3 tensor over image x image x orientation).
- Implements a variation of Prim's **Minimum Spanning Tree** algorithm with linked-hashmap based priority queue.

## Dependencies

- python 3.7+
- numpy 1.16+
- opencv-python 4.1.2+

## Execution Guide

### Quick Demo with Animation

```bash
pip install -r requirements.txt
bash demo.sh
```

### create_jigsaw_pieces.py

Reads and splits the image into rectangular jigsaw pieces, applying a random set of transformations.

```bash
create_jigsaw_pieces.py [OPTION] ${image_filename} ${x_slice} ${y_slice} ${keystring}
```

**Options:**
- `-v`: increase verbosity

![fragmentation demo](https://hj2choi.github.io/images/external/fragmentation_demo.JPG)

### solve_puzzle.py

Reconstructs the original image by assembling the puzzle pieces.

```bash
solve_puzzle.py [OPTION] ${keystring}
```

**Options:**
- `-v`: increase verbosity
- `-a`: show animation
- `-t`: show minimum spanning tree on top

![reconstruction demo](https://hj2choi.github.io/images/external/reconstruction_demo.JPG)

## Image Reconstruction Algorithm

```
I[i,t]: all puzzle pieces (image, transformation)
S[i,j,t]: all-pairs puzzle piece similarity-matrix in 3D tensor form
G[y,x]: puzzle block
Q: priority queue

1 initialize S with similarity metric
2 set all nodes in G as inactive
3 root <- set the root node with any image at position (0,0) (im: I[0,0], pos: (0,0))
4 Q.enqueue(root)
5 while Q is not empty do
6     v <- Q.dequeue()
7     G[v.pos].activate(v)
8     for all w, dir in G.inactiveNeighbors(v) do
9         w.im <- arg_max(S[v.im,j,k] for all j and k)
10        w.pos <- (v.pos+dir)
11        Q.enqueue(w)
12    Q.removeAllDuplicates(v.im)
```

## Time Complexity Analysis

**N**: number of images (puzzle pieces)  
**C**: total cache miss (total number of duplicate puzzle pieces to be removed from the queue)  
In all cases, **C = O(N)**

| Operations \ Algorithms | brute-force | brute-force<br>*(index mapping, hashmap)* | Prim's MST<br>*(max-heap)* | Prim's MST<br>*(linked-hashmap, matrix symmetry)* |
|:------------------------|:-----------:|:-----------------------------------------:|:--------------------------:|:--------------------------------------------------:|
| *similarity matrix* | *O(256N²)* | *O(32N²)* | *O(32N²)* | ***O(16N²)*** |
| traverse all puzzle pieces | O(N) | O(N) | O(N) | O(N) |
| traverse all positions | O(4N) | O(4N) | - | - |
| argmax(img at pos(x,y)) | O(256N) | O(32N) | O(32N) | O(32N) |
| validate puzzle-block shape | O(4N) | O(1) | O(1) | O(1) |
| *(PQueue)* remove by ID | - | O(C) | O(C log N) | **O(C)** |
| *(PQueue)* extract_min() | - | - | O(1) | **O(1)** |
| *(PQueue)* enqueue | - | - | **O(log N)** | O(N) |
| **Total time complexity** | *O(256N²)*<br>+ O(4096N⁴) | *O(32N²)*<br>+ O(32(C+N²))<br>+ O(128N³) | *O(32N²)*<br>+ O(32(C+N²))<br>+ O(3CN log N) | *O(16N²)*<br>+ O(32(C+N²))<br>+ O(N(C+N)) |
| **=** | O(N⁴) | O(N³) | O(N² log N) | **O(N²)** |

## References

- http://chenlab.ece.cornell.edu/people/Andy/publications/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf
- http://www.bmva.org/bmvc/2016/papers/paper139/paper139.pdf
- https://en.wikipedia.org/wiki/Prim%27s_algorithm
- https://en.wikipedia.org/wiki/Priority_queue
- https://github.com/python/cpython/blob/master/Lib/heapq.py
