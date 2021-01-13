# Jigsaw Puzzle Solver
slices image randomly and re-assemble them back to original image<br />
<br />

#### external dependencies
numpy, cv2

## Execution guide
### Quickstart: quick demo with animations
```sh
./run_automated_test_animated.sh
```
#### cut_image.py: slice and randomly transform image
```sh
cut_image.py ${image_file_name} ${x_slice} ${y_slice} ${output_filename_prefix} [OPTION]
```
[OPTION] -v: *enable console log*</br>
#### merge_image.py: re-assemble image fragments back to original image
```sh
merge_image.py ${input_filename_prefix} ${x_slice} ${y_slice} ${output_filename} [OPTION]
```
[OPTION] -v: *enable console log and show merge animation*<br/>


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
2. start with a empty 'board' and paste random image at center.<br />
3. while queue is not empty:<br />
&nbsp;&nbsp;&nbsp;&nbsp;3-1. for all possible 'board' locations (x,y) :<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3-1-1. find best-fit image and orientation pair I(x,y) = (img, orientation, score) at each position<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*3-1-2. (optimization) save best-fit image to cache*<br />
&nbsp;&nbsp;&nbsp;&nbsp;3-2. paste best-fit image, img = argmax(I(score)) to the 'board'<br />
&nbsp;&nbsp;&nbsp;&nbsp;3-3. remove image i from the queue<br />

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
