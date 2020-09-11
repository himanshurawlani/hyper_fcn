# Hyperparameter tuning with Keras and Ray Tune

This project uses HyperOpt's Bayesian optimization and Ray Tune to perform hyperparameter tuning for a simple image classifier. I explain the code and concepts used in this project in [this blogpost](https://medium.com/@himanshurawlani/hyperparameter-tuning-with-keras-and-ray-tune-1353e6586fda).

## Setup environment
Install the dependencies by running the command:
```
$ pip install -r requirements.txt
```

## Download the dataset
You can download the dataset using the [data.py](https://github.com/himanshurawlani/hyper_fcn/blob/master/data.py) script. You can also specifiy the number of images to kept in train and validation folder using the `--train-count` and `--val-count` flags:
```
$ python data.py --train-count 500 --val-count 100
```
This script downloads and extracts the data in the current working directory in `./dataset` folder.

## Check HyperOpt and Ray[tune] installation
Verify that hyperopt and ray is able to execute by running the following test script:
```
$ python test_hyperopt.py --smoke-test
```
Once the smoke test is successfull you can proceed to hyperparameter tuning using [train.py](https://github.com/himanshurawlani/hyper_fcn/blob/master/train.py)

## Start hyperparamter tuning
Before starting hyperparameter tuning process please update the [num_cpus](https://github.com/himanshurawlani/hyper_fcn/blob/master/train.py#L154) and [num_gpus](https://github.com/himanshurawlani/hyper_fcn/blob/master/train.py#L155) variables in [train.py](https://github.com/himanshurawlani/hyper_fcn/blob/master/train.py) according to your machine configuration. The default values are `num_cpus=2` and `num_gpus=0`. You can then start the hyperparameter tuning process by running the following command:
```
$ python train.py --train-dir dataset/train --val-dir dataset/val --optimize True --samples 100
```
If you do not want to run hyperparameter tuning but just want to train the image classifier using the default configuration (specified in [default_config.json](https://github.com/himanshurawlani/hyper_fcn/blob/master/default_config.json)) then you can use the following command:
```
$ python train.py --train-dir dataset/train --val-dir dataset/val
```

## Running inference on the trained model
Once the training is completed, the artifacts are stored in the `./snapshots` directory. You can run the inference by pass the `./test_images` folder to the following command:
```
$ python inference.py --test-dir ./test_images --snapshot-dir ./snapshots
```
The output of the inference would be a Pandas dataframe printed in the stdout


