# Fast Jigsaw Puzzle Solver with unknown orientation
- Breaks down an image into <b>N</b> (row x col) rectangular puzzle pieces of 8 possible random orientations.</br>
- Reconstructs <b>N</b> puzzle pieces back to original image in <b>O(N<sup>2</sup>)</b> runtime.</br>
![demo_anim](https://hj2choi.github.io/images/external/jigsaw_puzzle_solver_2.gif)</br>
<i>Disclaimer: orientation of the reconstructed image is random. Successful reconstruction is not always guaranteed.</i>

### Features
  - Prim's <b>Minimum Spanning Tree</b> algorithm with Linked-Hashmap implementation of the Priority Queue<br>
  - <b>Parallel</b> 3D distance matrix (img x img x orientation) computation  <br>
  - Euclidean distance metric for image boundary matching


### Dependencies
python 3.7+  
numpy 1.16+  
opencv-python 4.1.2+  

## Execution guide
### Quick demo with animation
```bash
pip install -r requirements.txt
bash demo.sh
```  

#### create_jigsaw_pieces.py: read and slice image into rectangular jigsaw pieces and apply random set of transformations
```bash
create_jigsaw_pieces.py [OPTION] ${image_filename} ${x_slice} ${y_slice} ${keystring}
```
-v: *increase verbosity*</br>
<img src="https://hj2choi.github.io/images/external/fragmentation_demo.JPG" width="560" title="image fragmentation visual demo">
</br>

#### solve_puzzle.py: reconstruct original image by putting puzzle pieces together
```bash
solve_puzzle.py [OPTION] ${keystring}
```
-v: *increase verbosity*<br/>
-a: *show animation*<br/>
-t: *show minimum spanning tree on top*<br/>
<img src="https://hj2choi.github.io/images/external/reconstruction_demo.JPG" width="560" title="reconstruction result">


## image reconstruction algorithm
```
I[i,t]: all puzzle pieces (image, transformation)
S[i,j,t]: all-pairs puzzle piece similarity-matrix
G[y,x]: puzzle block
Q: priority queue

1 initialize S with similarity metric
2 set all nodes in G as inactive
3 root <- (im: I[0,0], pos: (0,0))
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

## Time complexity analysis
<b>N</b>: number of images (puzzle pieces)</br>
<b>C</b>: total cache miss (total number of duplicate puzzle pieces to be removed from the queue)</br>
in all cases, <b>C = O(N)</b></br>

| Operations \ Algorithms       | brute-force<br><br><br> | brute-force</br><sub><sup><i>index mapping</i></br><i>hashmap</i></sub></sup> | Prim's MST</br><sub><sup><i>max-heap</i></sub></sup><br><br> | Prim's MST</br><sub><sup><i>linked-hashmap</i></sub></sup></br><sub><sup><i>matrix symmetry</i></sub></sup> |
|:------------------------------| :---: | :---: | :---: | :---: |
| <i>similarity matrix</i>      | <i>O(256N<sup>2</sup>) | <i>O(32N<sup>2</sup>) | <i>O(32N<sup>2</sup>) | <i><b>O(16N<sup>2</sup>)</b></i> |
| traverse all puzzle pieces    | O(N) | O(N) | O(N) | O(N) |
| traverse all positions        | O(4N) | O(4N) | - | - |
| argmax(img at pos(x,y))       | O(256N) | O(32N) | O(32N) | O(32N) |
| validate puzzle-block shape   | O(4N) | O(1) | O(1) | O(1) |
| <i>(PQueue)</i> remove by ID  | - | O(C) | O(ClogN) | <b>O(C)</b> |
| <i>(PQueue)</i> extract_min() | - | - | O(1) | <b>O(1)</b> |
| <i>(PQueue)</i> enqueue       | - | - | <b>O(logN)</b> | O(N) |
| <b>Total time complexity</b>  | <i>O(256N<sup>2</sup>)</i></br>+O(4096N<sup>4</sup>) | <i>O(32N<sup>2</sup>)</i></br>+O(32(C+N<sup>2</sup>))</br>+O(128N<sup>3</sup>) | <i>O(32N<sup>2</sup>)</i></br>+O(32(C+N<sup>2</sup>))</br>+O(3CNlog(N))</br> | <i>O(16N<sup>2</sup>)</i></br>+O(32(C+N<sup>2</sup>))</br>+O(N(C+N)) |
| <b>=</b>                      | O(N<sup>4</sup>) | O(N<sup>3</sup>) | O(N<sup>2</sup>log(N)) | <b>O(N<sup>2</sup>)</b> |

### references
http://chenlab.ece.cornell.edu/people/Andy/publications/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf</br>
http://www.bmva.org/bmvc/2016/papers/paper139/paper139.pdf</br>
https://en.wikipedia.org/wiki/Prim%27s_algorithm</br>
https://en.wikipedia.org/wiki/Priority_queue</br>
https://github.com/python/cpython/blob/master/Lib/heapq.py</br>
