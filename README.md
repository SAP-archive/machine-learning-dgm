# Learning to Remember what to Remember: A Synaptic Plasticity Driven Framework for Continual Learning

## Description: 
A continual learning framework for class incremental learning described in the following paper [arXiv](https://arxiv.org/abs/1904.03137).
Note, this is work in progress and this code that will be dynamically updated.

This repository currently contains code to run experiments of DGMw on three datasets: MIST, SVHN, ImageNet.
## Requirements

Please, find a list with requiered packages and versions in [requierements.txt](https://github.com/SAP/machine-learning-dgm/blob/master/requierements.txt) file.

## Download and Installation

TODO

## Known Issues
No issues known

## How to obtain support
This project is provided "as-is" and any bug reports are not guaranteed to be fixed.

## Running the tests
In orer to start experiemtns, run the script passing the dataset name as argument (mnist/svhn):
```
python run.py --dataset mnist --method DGMw
```
Please, change the metaparmeters in the corresponding file [cfg/](https://github.com/SAP/machine-learning-dgm/tree/master/cfg) if needed.

To run on the ImageNet dataset use the [run_DGMw_imagenet.py/](https://github.com/SAP/machine-learning-dgm/tree/master/run_DGMw_imagenet.py) script.
## License

This project is licensed under SAP Sample Code License Agreement except as noted otherwise in the [LICENSE file](LICENSE.md).

