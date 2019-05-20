import os
from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf

#model_file = "model.json"
#weights_file = "weights.h5"
TF_MODEL_FILES = "./TF_Model/tf_model"

def keras2tf(model, weights_file = None):

    if weights_file:
        with open(model, "r") as file:
            config = file.read()

        model = model_from_json(config)
        model.load_weights(weights_file)

    saver = tf.train.Saver()
    with K.get_session() as sess:
        K.set_learning_phase(0)
        saver.save(sess, TF_MODEL_FILES)

    fw = tf.summary.FileWriter('logs', sess.graph)
    fw.close()

def compile_movidius_graph(tf_model_path, input_node, output_node, graph_path):
    os.system('mvNCCompile {0}.meta -in {1} -on {2} -o {3}'.format(tf_model_path, input_node, output_node, graph_path))

def save_keras_model(model, model_filename = "model.json", weights_filename = "weights.h5"):
    with open(model_filename, "w") as file:
        file.write(model.to_json())
    model.save_weights(weights_filename)