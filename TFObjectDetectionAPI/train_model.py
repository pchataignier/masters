import argparse
import tensorflow as tf
from object_detection import model_lib
from object_detection import model_hparams

parser = argparse.ArgumentParser(description="Train model from pipeline configs")
parser.add_argument('-o', '--out_dir', required=True,
                    help="Path to output model directory where event and checkpoint files will be written.")
parser.add_argument('-p', '--pipeline_config', required=True, help="Path to pipeline config file.")
parser.add_argument('--num_train_steps', required=False, default=None, type=int, help="Number of training steps (overrides pipeline).")
parser.add_argument('--sample_1_of_n_eval_examples', required=False, default=1, type=int, help="Will sample one of every n eval input examples, where n is provided.")

args = parser.parse_args()

config = tf.estimator.RunConfig(model_dir=args.out_dir)

train_and_eval_dict = model_lib.create_estimator_and_inputs(
    run_config=config,
    pipeline_config_path=args.pipeline_config,
    train_steps=args.num_train_steps,
    sample_1_of_n_eval_examples=args.sample_1_of_n_eval_examples,
    hparams=model_hparams.create_hparams(None))

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