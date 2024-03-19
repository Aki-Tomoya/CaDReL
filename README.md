# CaDReL
The official repository of CaDReL.
![](https://github.com/Aki-Tomoya/CaDReL/blob/main/fig2.png)
## Environment setup
        pytorch==1.8.2+cu111
        python==3.9.16

Clone the repository and create the `camel` using:

        conda env create -n camel
        conda activate camel
    

Then download spacy data by executing the following command:

        python -m spacy download en

## Data Preparation
To run the code, annotations and visual features for the COCO dataset are needed.Please download the zip files containing the images [train2014.zip](http://images.cocodataset.org/zips/train2014.zip) [val2014](http://images.cocodataset.org/zips/val2014.zip) and the annotations file[annotations.zip](https://pan.baidu.com/s/17ik-2OZGFaQ5-AzCCWkL9w).To reproduce our result, please generate the corresponding feature files (`COCO2014_RN50x16.hdf5`)using the code in the folder.

## Evaluation
To evaluate the results of your model, you can use `test.py` in `CaDReL/Camel/utils` folder.
To evaluate the models on coco online test, you can use `online_test.py`

## Training procedure
Run python `run.py` to train your model. Args saved in main.py.

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name (default: `CaDReL`) |
| `--batch_size` | Batch size (default: `65`) |
| `--workers` | Number of workers (default: `0`) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint |
| `--resume_best` | If used, the training will be resumed from the best checkpoint |
| `--annotation_folder` | Path to folder with COCO annotations (required) |
| `--image_folder` | Path to folder with COCO images (required) |
| `--clip_variant` | CLIP variant to be used as image encoder (default: `RN50x16`) |
| `--distillation_weight` | Weight for the knowledge distillation loss (default: `0.1` in XE phase, `0.005` in SCST phase) |
| `--ema_weight` | Target decay rate of Mean Teacher paradigm (default: `0.999`) |
| `--phase` | Training phase, `xe` or `scst` (default: `xe`) |
| `--disable_mesh` | If used, the model does not employ the mesh connectivity |
| `--saved_model_file` | If used, path to model weights to be loaded |
| `--N_dec` | Number of decoder layers (default: `3`) |
| `--N_enc` | Number of encoder layers (default: `3`) |
| `--d_model` | Dimensionality of the model (default: `512`) |
| `--d_ff` | Dimensionality of Feed-Forward layers (default: `2048`) |
| `--m` | Number of memory vectors (default: `40`) |
| `--head` | Number of heads (default: `8`) |
| `--warmup` | Warmup value for learning rate scheduling (default: `10000`) |
