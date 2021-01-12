# Jigsaw Puzzle Solver
slices image randomly and re-assemble them back to original image




### external dependencies
numpy
cv2

## Execution guide
#### slice and randomly transform image
```sh
cut_image.py ${image_file_name} ${x_slice} ${y_slice} ${output_filename_prefix}
```
#### re-assemble image fragments back to original image
```sh
merge_image.py ${input_filename_prefix} ${x_slice} ${y_slice} ${output_filename}
```
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
| `enable_merge_visualization` | enable step by step merge animation |
| `animation_interval_millis` | milliseconds interval between each merge step in animation |

## image assembly algorithm
1. place all images in a queue.
  2. start with a empty 'board' and paste random image at (r=0, c=0)
  2. while queue is not empty:
      2-1. for all possible position for any image to be pasted:
          2-1-1. find best-fit image, and its transformation and location
                  in accordance to similarity matrix
          2-1-2. save local best-fit image to cache
      2-2. paste best-fit image to the 'board'
      2-3. remove image from queue
  3. construct final image

## optimization techniques
- preprocessed all-pairs image border similarity metric
- used cache mechanism to minimize redundant computation.
- used parallel processing (multi-processing) when heavy load of computation is required
- used index mapping table for looking up similarity score for each image orientation and stitching directions

## TODO
- minimize I/O overhead when forking processes. Need to go around python's Global Interpreter Lock.
- performance drastically decreases when fragmented images are square shaped. Need to compute 256-dimension orientation mapping table
- exploiting diagonality property of distance matrix might open a room for further optimization

### references
http://chenlab.ece.cornell.edu/people/Andy/publications/Andy_files/Gallagher_cvpr2012_puzzleAssembly.pdf
http://www.bmva.org/bmvc/2016/papers/paper139/paper139.pdf
