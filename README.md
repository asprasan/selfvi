# SeLFVi

## Requirements
```
pytorch >= 1.6.0
torchvision >= 0.7.0
tensorboardX >= 2.4
numpy >= 1.18.1
```

## Usage

### For training
- **Preparing training data**:
The dataloader associated with the training code reads a sequence of stereo frames from a _hdf5_ file.
The shape of the training data array should be: (number_of_videos, frames_per_video, stereo_views, RGB, height, width). An example data array would have the shape `[200, 5, 2, 3, 100, 100]`, where the data consists of `200` videos, each with `5` stereo RGB image patches of size `100x100`.
The dataloader inherently splits this whole training data into a training and validation dataset.
The percentage of validation data set aside can be controlled using the parameter `--val` during training.
The data in hdf5 file requires a `key` to access it. In the given code, the key is assumed to be `'train'`. This can be changed in the `data.py` file, if necessary.
The hdf5 file can be stored in the `data` directory and the path can be specified using the parameters `data-path` and `h5-file`.
- **Training**:
After the necessary environment is set, one can train the network by simply running `sh train.sh`.
For information on the parameters used for training please run `python train.py --help`.



### For quantitative evaluation

- **Preparing test data**: Store your test data in .h5 (also known as hdf5) format into the `data` directory. 
The code expects ground truth data of size 7x7 LF. 
The shape of the data resembles something like: (15, 5, 49, 3, 352, 512) which is (number_of_videos, frames_per_video, angular_views, RGB, height, width). While saving the data into the h5 file, one generally uses a dictionary. The `key` name to be used for the dictionary is `'test'`. So that the data should be accessible when you read `h5_file['test']`.
- The checkpoint file should be copied into the `weights` directory. Download the checkpoint file from [here](https://drive.google.com/file/d/1f_W5-2vdXQPmBohKkRpKPIk2TKGKcPHG/view?usp=sharing).
- In the `test_lf.sh` file, insert values for `h5-file`, `inph`, `inpw`.
- Then run the `test_lf.sh` file. The results will be saved in the `results` directory.

### For qualitative evaluation
We also provide a jupyter notebook to evaluate our network on some stereo video data used in our paper. The stereo data does not belong to us and is taken from [this paper](https://ieeexplore.ieee.org/abstract/document/6263847) (Please cite the corresponding paper if you use their data).
Just like the training data, a sequence of stereo frames can be used to generate LF videos using this code.
The default code requires the test data in a hdf5 file where the data is in the format (number_of_videos, frames_per_video, stereo_views, RGB, height, width) (e.g. `[30, 5, 2, 3, 200, 200]`).
A sample stereo data which is a pre-processed version of the raw dataset can be downloaded from [here](https://drive.google.com/file/d/1nMAm7HD6Dy80d9HZcZTSm0pEVgxRAXK6/view?usp=sharing).



### Some guidelines for preparing data for best performance
Our model is trained on synthesized stereo videos from LF images. So, it works best when:
- the disparity between stereo frames is quite low. I would say that the disparity between -3 and +3 pixels.
- the zero-disparity plane is between the nearest the farthest object. So, that there's both negative and positve disparities in the stereo pair.