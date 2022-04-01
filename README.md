# Fast Jigsaw Puzzle Solver with unknown orientation
- Fragments image into <b>N</b> (row x col) slices of random orientations</br>
- Re-assemble <b>N</b> image fragments back to original image in <b>O(N<sup>2</sup>)</b> time complexity</br>
![demo_anim](https://hj2choi.github.io/images/external/jigsaw_puzzle_solver.gif)</br>
<i>Disclaimer: successful reconstruction is not guaranteed</i>

### Features
  - distance-matrix computation with <b>parallel processing</b><br>
  - Prim's <b>Minimum Spanning Tree</b> algorithm with <b>Linked-Hashmap</b> implementation of Priority Queue<br>


### Dependencies
python 3.9.1<br>
numpy 1.20.2<br>
opencv-python 4.5.1.48

## Execution guide
### Quick demo with animation
```bash
pip install -r requirements.txt
bash demo.sh
```

#### fragment_image.py: slice and randomly transform image
```bash
fragment_image.py ${image_file_name} ${x_slice} ${y_slice} ${output_filename_prefix} [-v]
```
-v: *output log*</br>
<img src="https://hj2choi.github.io/images/external/cut_image.png" width="300" title="fragment image">
</br>

#### assemble_fragments.py: re-assemble image fragments back to original image
```bash
assemble_fragments.py ${input_filename_prefix} ${x_slice} ${y_slice} ${output_filename} [-v]
```
-v: *output log and show animation*<br/>
<img src="https://hj2choi.github.io/images/external/merge_image.png" width="300" title="merge image">

## config.ini
| Key | Description | Default |
| :--- | --- | --- |
| `fragments_dir` | directory to save fragmented images | image_fragments/ |
| `output_dir` | directory to save final merged image | images_out/ |
| `debug` | enable console log | False |
| `enable_visualization` | show assembly animation after the process is complete | False |
| `animation_interval_millis` | milliseconds interval between each merge step in animation | 100 |

## image assembly algorithm (modified Prim's Minimum Spanning Tree)
```
I[i,t]: all image fragments (image, transformation)
S[i,j,t]: all-pairs image similarity-matrix
G[y,x]: image cell-block
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
<b>N</b>: number of images</br>
<b>C</b>: total cache miss (number of duplicate images to be removed from queue)</br>
in all cases, <b>C = O(N)</b></br>

| Operations \ Algorithms | brute-force<br><br><br> | brute-force</br><sub><sup><i>index mapping</i></br><i>hashmap</i></sub></sup> | Prim's MST</br><sub><sup><i>max-heap</i></sub></sup><br><br> | Prim's MST</br><sub><sup><i>linked-hashmap</i></sub></sup></br><sub><sup><i>matrix symmetry</i></sub></sup> |
| :---------------------------- | :---: | :---: | :---: | :---: |
| <i>similarity matrix</i>      | <i>O(256N<sup>2</sup>) | <i>O(32N<sup>2</sup>) | <i>O(32N<sup>2</sup>) | <i><b>O(16N<sup>2</sup>)</b></i> |
| traverse all images           | O(N) | O(N) | O(N) | O(N) |
| traverse all positions        | O(4N) | O(4N) | - | - |
| argmax(img at pos(x,y))       | O(256N) | <b>O(32N)</b> | O(32N) | O(32N) |
| validate cellblock shape      | O(4N) | <b>O(1)</b> | O(1) | O(1) |
| <i>(PQueue)</i> remove by ID  | - | O(C) | O(ClogN) | <b>O(C)</b> |
| <i>(PQueue)</i> extract_min() | - | - | O(logN) | <b>O(1)</b> |
| <i>(PQueue)</i> enqueue       | - | - | <b>O(logN)</b> | O(N) |
| <b>Total time complexity</b> | <i>O(256N<sup>2</sup>)</i></br>+O(4096N<sup>4</sup>) | <i>O(32N<sup>2</sup>)</i></br>+O(32(C+N<sup>2</sup>))</br>+O(128N<sup>3</sup>) | <i>O(32N<sup>2</sup>)</i></br>+O(32(C+N<sup>2</sup>))</br>+O(3CNlog(N))</br> | <i>O(16N<sup>2</sup>)</i></br>+O(32(C+N<sup>2</sup>))</br>+O(N(C+N)) |
| <b>=</b>  | O(N<sup>4</sup>) | O(N<sup>3</sup>) | O(N<sup>2</sup>log(N)) | <b>O(N<sup>2</sup>)</b> |

### references
http://chenlab.ece.cornell.edu/people/Andy/publications/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf</br>
http://www.bmva.org/bmvc/2016/papers/paper139/paper139.pdf</br>
https://en.wikipedia.org/wiki/Prim%27s_algorithm</br>
https://en.wikipedia.org/wiki/Priority_queue</br>
https://github.com/python/cpython/blob/master/Lib/heapq.py</br>
