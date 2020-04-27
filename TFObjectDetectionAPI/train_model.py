import os
import argparse
import tensorflow as tf
from datetime import datetime
from object_detection import model_lib
from object_detection import model_hparams
from object_detection.protos import pipeline_pb2

def tensorflow_shutup(verbose=False):
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        if not verbose:
            #import os
            from tensorflow.compat.v1 import logging
            logging.set_verbosity(logging.ERROR)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass

def find_latest_checkpoint(log_dir, modelId):
    from glob import glob

    log_dir = os.path.normpath(log_dir)
    runs = glob(f"{log_dir}/{modelId}*")

    if not runs:
        return None

    runs.sort()
    latest_run = runs[-1]

    checkpoint = tf.train.get_checkpoint_state(latest_run)
    return checkpoint.model_checkpoint_path if checkpoint else None


parser = argparse.ArgumentParser(description="Train model from pipeline configs")
parser.add_argument('-o', '--out_dir', required=True,
                    help="Path to output model directory where event and checkpoint files will be written.")
parser.add_argument('-p', '--pipeline_config', required=True, help="Path to pipeline config file.")
parser.add_argument('--num_train_steps', required=False, default=None, type=int,
                    help="Number of training steps (overrides pipeline).")
parser.add_argument('--max_checkpoints', required=False, default=None, type=int,
                    help="Maximum checkpoints to save. Default: no limit")
parser.add_argument('--checkpoint_steps', required=False, default=1000, type=int,
                    help="Save checkpoint at every x steps. Default: every 1000 steps")
parser.add_argument('--sample_1_of_n_eval_examples', required=False, default=1, type=int,
                    help="Will sample one of every n eval input examples, where n is provided.")
parser.add_argument('-l', '--logs', required=False, help="Path to logs directory")
parser.add_argument('-v', '--verbose', required=False, action='store_true')

args = parser.parse_args()

tensorflow_shutup(args.verbose)

timestamp = datetime.now().strftime("-%Y%m%d-%H%M%S-%f")
modelId = os.path.basename(os.path.normpath(args.out_dir))
out_dir = args.out_dir+timestamp
log_dir = args.logs if args.logs else out_dir

config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config = tf.estimator.RunConfig(model_dir=out_dir, session_config=config_proto,
                                save_checkpoints_steps=args.checkpoint_steps, keep_checkpoint_max=args.max_checkpoints)

pipeline_overrides = None
latest_checkpoint = find_latest_checkpoint(log_dir, modelId)
if latest_checkpoint:
    pipeline_overrides = pipeline_pb2.TrainEvalPipelineConfig()
    pipeline_overrides.train_config.fine_tune_checkpoint = latest_checkpoint

train_and_eval_dict = model_lib.create_estimator_and_inputs(
    run_config=config,
    pipeline_config_path=args.pipeline_config,
    config_override=pipeline_overrides,
    train_steps=args.num_train_steps,
    sample_1_of_n_eval_examples=args.sample_1_of_n_eval_examples,
    hparams=model_hparams.create_hparams(None),
    save_final_config=True)

estimator = train_and_eval_dict['estimator']
train_input_fn = train_and_eval_dict['train_input_fn']
eval_input_fns = train_and_eval_dict['eval_input_fns']
eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
predict_input_fn = train_and_eval_dict['predict_input_fn']
train_steps = train_and_eval_dict['train_steps']

train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

# Log that training completed
with open(os.path.join(log_dir, "completed.log"), 'a') as f:
    f.write(f"{modelId},{timestamp}\n")