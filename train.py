
import os
import sys
import json
import traceback
import argparse
import time

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from hyperopt import hp

from model import FCN_model
from generator import Generator
from callbacks import create_callbacks
import logger as Logger

logger = Logger.get_logger('hyper_fcn', './logs/training')

class Trainable:
    def __init__(self, train_dir, val_dir, snapshot_dir, final_run=False):
        # Initializing state variables for the run
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.final_run = final_run
        self.snapshot_dir = snapshot_dir

    def dump_classes(self, classes):
        with open(os.path.join(self.snapshot_dir, 'classes.txt'), 'w') as f:
            for class_name in classes:
                print(class_name, file=f)

    def train(self, config, reporter=None):
        # As of 10/12/2019: One caveat of using TF2.0 is that TF AutoGraph
        # functionality does not interact nicely with Ray actors. One way to get around
        # this is to `import tensorflow` inside the Tune Trainable.
        import tensorflow as tf

        # If you get out of memory error try reducing the batch size
        train_generator = Generator(self.train_dir, config['batch_size'])
        val_generator = Generator(self.val_dir, config['batch_size'])

        # Save class names to a text file to be used at inference
        if self.final_run:
            self.dump_classes(train_generator.classes)

        # Create FCN model
        model = FCN_model(config, len_classes=len(train_generator.classes))

        # Compile model with losses and metrics
        model.compile(optimizer=tf.keras.optimizers.Nadam(lr=config['lr']),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        
        # Create callbacks to be used during model training
        callbacks = create_callbacks(self.final_run, self.snapshot_dir)

        logger.info("Starting model training")
        # Start model training
        history = model.fit(train_generator,
                            # steps_per_epoch=len(train_generator),
                            steps_per_epoch=1,
                            epochs=1,
                            callbacks=callbacks,
                            validation_data=val_generator,
                            # validation_steps=len(val_generator)
                            validation_steps=1
                            )

        return history


def create_search_space():
    # Creating hyperopt search space
    search_space = {"lr": hp.choice("lr", [0.0001, 0.001, 0.01, 0.1]),
                    "batch_size": hp.choice("batch_size", [8, 16, 32, 64]), 
                    "use_contrast": hp.choice("use_contrast", ["True", "False"]),
                    "contrast_factor": hp.choice('contrast_factor', [0.1, 0.2, 0.3, 0.4]),
                    "use_rotation": hp.choice("use_rotation", ["True", "False"]),
                    "rotation_factor": hp.choice('rotation_factor', [0.1, 0.2, 0.3, 0.4]),
                    "use_flip": hp.choice("use_flip", ["True", "False"]),
                    "flip_mode": hp.choice('flip_mode', ["horizontal", "vertical"]),
                    "dropout_rate": hp.choice("dropout_rate", [0.1, 0.2, 0.3, 0.4, 0.5]),
                    "conv_block1_filters":hp.choice("conv_block1_filters", [32, 64, 128, 256, 512]),
                    "conv_block2_filters":hp.choice("conv_block2_filters", [32, 64, 128, 256, 512]),
                    "conv_block3_filters":hp.choice("conv_block3_filters", [32, 64, 128, 256, 512]),
                    "conv_block4_filters":hp.choice("conv_block4_filters", [32, 64, 128, 256, 512]),
                    "conv_block5_filters":hp.choice("conv_block5_filters", [32, 64, 128, 256, 512]),
                    "fc_layer_type": hp.choice("fc_layer_type", ['dense', 'convolution']),
                    "pool_type": hp.choice("pool_type", ['max', 'average']),
                    "fc1_units":hp.choice("fc1_units", [32, 64, 128, 256, 512])}

    # Current best setting
    # For hp.uniform specify the exact value
    # For hp.choice specify the index (0 based indexing) in the array
    intial_best_config = [{"lr": 0,
                            "batch_size": 0, 
                            "use_contrast": 1,
                            "contrast_factor": 0,
                            "use_rotation": 1,
                            "rotation_factor": 0,
                            "use_flip": 1,
                            "flip_mode": 0,
                            "dropout_rate": 1,
                            "conv_block1_filters": 0,
                            "conv_block2_filters": 1,
                            "conv_block3_filters": 2,
                            "conv_block4_filters": 3,
                            "conv_block5_filters": 4,
                            "fc_layer_type": 1,
                            "pool_type": 0,
                            "fc1_units": 1}]

    return search_space, intial_best_config

def parse_args(args):
    """
    Example command:
    $ python train.py --train-dir dataset/train --val-dir dataset/val --optimize True --samples 100
    """
    parser = argparse.ArgumentParser(description='Optimize RetinaNet anchor configuration')
    parser.add_argument('--train-dir', type=str, help='Path to training directory containing folders with images for each class.')
    parser.add_argument('--val-dir', type=str, help='Path to validation directory containing folders with images for each class.')
    parser.add_argument('--snapshot-dir', type=str, help='Path to validation directory containing folders with images for each class.',
                                             default='./snapshots')
    parser.add_argument('--config-path', type=str, help='FCN model config path (considered only when optimize=False).', default='./default_config.json')
    parser.add_argument('--optimize', type=str, help='Flag to run hyperparameter search.', default="False")
    parser.add_argument('--samples', type=int, help='Number of times to sample from the hyperparameter space.', default=64)

    return parser.parse_args(args)

def main(args=None):

    # Parse command line arguments.
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Create snapshot directory
    os.makedirs(args.snapshot_dir, exist_ok=True)

    if args.optimize == "True":
        logger.info("Initializing ray")
        ray.init(configure_logging=False)

        logger.info("Initializing ray search space")
        search_space, intial_best_config = create_search_space()

        # TODO: Adapt the below parameters according to the machine configuration
        num_samples = args.samples
        num_cpus = 2
        num_gpus = 0

        logger.info("Initializing scheduler and search algorithms")
        # Use HyperBand scheduler to earlystop unpromising runs
        scheduler = AsyncHyperBandScheduler(time_attr='training_iteration',
                                            metric="val_loss",
                                            mode="min",
                                            grace_period=10)

        # Use bayesian optimisation provided by hyperopt
        search_alg = HyperOptSearch(search_space,
                                    metric="val_loss",
                                    mode="min",
                                    points_to_evaluate=intial_best_config)

        # We limit concurrent trials to 1 since bayesian optimisation doesn't parallelize very well
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)
        
        logger.info("Initializing ray Trainable")
        # Initialize Trainable for hyperparameter tuning
        trainer = Trainable(os.path.abspath(args.train_dir), 
                            os.path.abspath(args.val_dir), 
                            os.path.abspath(args.snapshot_dir), 
                            final_run=False)

        logger.info("Starting hyperparameter tuning")
        analysis = tune.run(trainer.train, 
                            verbose=1, 
                            num_samples=num_samples,
                            search_alg=search_alg,
                            scheduler=scheduler,
                            raise_on_failed_trial=False,
                            resources_per_trial={"cpu": num_cpus, "gpu": num_gpus}
                            )

        best_config = analysis.get_best_config(metric="val_loss", mode='min')
        logger.info(f'Best config: {best_config}')

        if best_config is None:
            logger.error(f'Optimization failed')
        else:
            logger.info("Saving best model config")
            with open(os.path.join(args.snapshot_dir, 'config.json'), 'w') as f:
                json.dump(best_config, f, indent=4)

            logger.info("Waiting for GPU/CPU memory cleanup")
            time.sleep(3)

            logger.info(f"Refitting the model on best config")
            trainer = Trainable(args.train_dir, args.val_dir, args.snapshot_dir, final_run=True)
            history = trainer.train(best_config, reporter=None)
    else:
        with open(args.config_path, 'r') as f:
            default_config = json.load(f)
        with open(os.path.join(args.snapshot_dir, 'config.json'), 'w') as f:
            json.dump(default_config, f, indent=4)
        logger.info(f"Training the model on default config")
        trainer = Trainable(args.train_dir, args.val_dir, args.snapshot_dir, final_run=True)
        history = trainer.train(default_config, reporter=None)

    logger.info("Training completed")

if __name__ == "__main__":
    try:
        main()
    except:
        logger.error(traceback.format_exc())
        raise
