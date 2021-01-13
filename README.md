# Jigsaw Puzzle Solver
slices image randomly and re-assemble them back to original image<br />
<br /><br /><br />

### external dependencies
numpy,<br />
cv2

## Execution guide
#### slice and randomly transform image
```sh
cut_image.py ${image_file_name} ${x_slice} ${y_slice} ${output_filename_prefix} [OPTION]
```
OPTION -v: enable console log</br>
#### re-assemble image fragments back to original image
```sh
merge_image.py ${input_filename_prefix} ${x_slice} ${y_slice} ${output_filename} [OPTION]
```
OPTION -v: enable console log and show merge animation<br/>
#### quick testrun with animations
```sh
./run_automated_test.sh
```

## config.ini
| Key | Description |
| :---: | --- |
| `images_dir` | directory to save fragmented (sliced and randomly transformed) images |
| `output_dir` | directory to save final merged image |
| `debug` | enable console logging |
| `enable_merge_visualization` | show merging animation |
| `animation_interval_millis` | milliseconds interval between each merge step in animation |

## image assembly algorithm
1. place all images in a queue.<br />
2. start with a empty 'board' and paste random image at (r=0, c=0)<br />
3. while queue is not empty:<br />
&nbsp;&nbsp;&nbsp;&nbsp;3-1. for all possible position for any image to be pasted:<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3-1-1. find best-fit image, and its transformation and location in accordance to similarity matrix<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3-1-2. save local best-fit image to cache<br />
&nbsp;&nbsp;&nbsp;&nbsp;3-2. paste best-fit image to the 'board'<br />
&nbsp;&nbsp;&nbsp;&nbsp;3-3. remove image from queue<br />
4. construct final image<br />

## optimization techniques
- preprocessed all-pairs image border similarity metric
- used cache mechanism to minimize redundant computation.
- used parallel processing (multi-processing) when heavy load of computation is required
- used index mapping table for looking up similarity score for each orientation and stitching directions

## TODO
- too much I/O overhead when creating process. Need to do something about python's Global Interpreter Lock.
- exploiting diagonality property of distance matrix might open a room for further optimization

### references
http://chenlab.ece.cornell.edu/people/Andy/publications/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf
http://www.bmva.org/bmvc/2016/papers/paper139/paper139.pdf
