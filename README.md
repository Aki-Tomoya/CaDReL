# CaDReL
The official repository of CaDReL.
## Environment setup
pytorch==1.8.2+cu111
python==3.9.16

Clone the repository and create the `camel` using 
    conda env create -n camel
    conda activate camel
    

Then download spacy data by executing the following command:
    python -m spacy download en

## Data Preparation
To run the code, annotations and visual features for the COCO dataset are needed. Please download the annotations file[annotations.zip](https://pan.baidu.com/s/17ik-2OZGFaQ5-AzCCWkL9w).To reproduce our result, please generate the corresponding feature files (`COCO2014_RN50x16.hdf5`)using the code in the folder.

## Evaluation
To evaluate the results of your model, you can use `test.py` in `CaDReL/Camel/utils` folder.
To evaluate the models on coco online test, you can use `online_test.py`

## Training procedure
Run python `run.py` to train your model.
