# Soft Decision Tree

An implementation of Frosst & Hinton's 
[Distilling a Neural Network Into a Soft Decision Tree](http://ceur-ws.org/Vol-2071/CExAIIA_2017_paper_3.pdf)

## Requirements

The project was developed using Python 3.6 and uses the following libraries: 
- PyTorch 1.3.1
- Torchvision 0.4.2
- Tensorboard 1.15.0

One can install the required libraries by using `pip` in a Conda environment or virtualenv with the provided 
`requirements.txt` file:

`pip install -r requirements.txt`

## Usage

`main.py` will fit a Soft Decision Tree on the MNIST dataset, acquired through Torchvision.
One can simply run the script with default arguments with the following command:
`python main.py`

Please refer to `args.py` to retrieve all the available arguments and alter to your liking :)
