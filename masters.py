# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 13:28:08 2019
BG - 0
Telefone - 1
Caneca - 2
Controle Remoto - 3
Garrafa - 4
Mão - 5
Base de Telefone(?)

@author: Pedro Schuback Chataignier
"""
import os
import numpy as np
import imgaug
import json
import random
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from PIL import Image, ImageDraw


MODEL_NAME = "masters" # TODO: Model name
CLASSES = ["Telefone", "Caneca", "Controle Remoto", "Garrafa", "Mão"]

ROOT_DIR = os.getcwd() # Root directory of the project
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs") # Default directory to save logs and model checkpoints
DEFAULT_CONFIGS_DIR = os.path.join(ROOT_DIR, "configs") # Default directory to configuration files

############################################################
#  Configurations
############################################################

class MastersConfig(Config):
    """Configuration for training on custom Dataset.
    Derives from the base Config class and overrides values for specific dataset."""
    
    # Give the configuration a recognizable name
    NAME = MODEL_NAME

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CLASSES)

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    # !! Set 'IMAGES_PER_GPU' property on config file

    # NUMBER OF GPUs to use. For CPU training, use 1
    # !! Set 'GPU_COUNT' property on config file

    def __init__(self, configFile = None):
        super().__init__()

        if configFile:
            configs = json.load(open(configFile))

            for atrib in configs:
                setattr(self, atrib, configs[atrib])

class InferenceConfig(MastersConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    def __init__(self, configFile = None):
        super().__init__(configFile)
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.BATCH_SIZE = 1
        self.DETECTION_MIN_CONFIDENCE = 0

############################################################
#  Dataset
############################################################

class MastersDataset(utils.Dataset):
    def __init__(self):
        super().__init__()
        # Add classes
        for i in range(len(CLASSES)):
            self.add_class(MODEL_NAME, i + 1, CLASSES[i])

    @staticmethod
    def auto_split_validation(dataset_filepath, val_percent): #TODO: Checar val_percent > 100 e > 1
        dataset_dir = os.path.dirname(dataset_filepath)

        with open(os.path.normpath(dataset_filepath)) as file:
            dataset = json.load(file)

        images = list(dataset)
        random.shuffle(images)

        split_point = round((1-val_percent)*len(images))
        if split_point < 1:
            split_point = 1

        train_imgs = images[:split_point]
        training_dataset = MastersDataset()

        for i, img in enumerate(train_imgs):
            a=dataset[img]

            rel_path = a['relativePath'].replace("\\","/")
            imgPath = os.path.join(dataset_dir, rel_path)
            imgPath = os.path.normpath(imgPath)

            training_dataset.add_image(
                MODEL_NAME, image_id=i,  # TODO: id diferente? Talvez a própria chave 'img'
                path=imgPath,
                width=a['file_attributes']['Width'],
                height=a['file_attributes']['Height'],
                annotations=a)


        training_dataset.prepare()

        val_imgs = images[split_point:]
        validation_dataset = MastersDataset()

        for i, img in enumerate(val_imgs):
            a = dataset[img]

            rel_path = a['relativePath'].replace("\\", "/")
            imgPath = os.path.join(dataset_dir, rel_path)
            imgPath = os.path.normpath(imgPath)

            validation_dataset.add_image(
                MODEL_NAME, image_id=i,  # TODO: id diferente?
                path=imgPath,
                width=a['file_attributes']['Width'],
                height=a['file_attributes']['Height'],
                annotations=a)

        validation_dataset.prepare()

        return training_dataset, validation_dataset

    # New loading method.
    def load_masters(self, annotations_filepath): #TODO: remover ou alterar
        dataset_dir = os.path.dirname(annotations_filepath)

        # Add images
        with open(annotations_filepath) as file:
            annotations = json.load(file)

        for i, a in enumerate(annotations.values()):
            self.add_image(
                MODEL_NAME, image_id=i, #TODO: id diferente?
                path=os.path.join(dataset_dir, a['filename']),
                width=a['file_attributes']['Width'],
                height=a['file_attributes']['Height'],
                annotations=a)
    
    # Override from original Class
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If image source is unknown, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info['source'] != MODEL_NAME:
            return super(MastersDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]['annotations']
        
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for region in annotations['regions'].values():
            x = region['shape_attributes']['all_points_x']
            y = region['shape_attributes']['all_points_y']
            poly = list(zip(x,y)) #TODO: suspeito, ficar de olho            
            mask = self.polygon_to_mask(poly, image_info['width'], image_info['height'])
            
            class_ids.append(int(region['region_attributes']['Class']))
            instance_masks.append(mask)        

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(MastersDataset, self).load_mask(image_id)
    
    # Auxiliary function for 'load_mask'
    @staticmethod
    def polygon_to_mask(polygon, width, height):
        # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
    
        img = Image.new('L', (width, height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask = np.array(img)
        
        return mask
    
    # Override from original Class
    def image_reference(self, image_id):
        """Return the image's Filename."""
        info = self.image_info[image_id]
        if info["source"] == MODEL_NAME:
            return info["path"] #TODO: nome do arquivo?
        else:
            super(MastersDataset, self).image_reference(image_id)

############################################################
# Auxiliary Functions and Classes
############################################################



def GetConfig(mode, configFile=None):
    if mode == "train":
        return MastersConfig(configFile)
    else:
        return InferenceConfig(configFile)

def GetModel(mode, weights_path, logs_path, config, docker=False):
    if mode == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logs_path)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logs_path)

    # Select weights file to load
    if weights_path.lower() == "last":  # Find last trained weights
        model_path = model.find_last()[1]
    elif docker:
        import dockerUtils
        model_path = dockerUtils.GetModelPath(weights_path)
    elif weights_path.lower() == "imagenet": # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()

    else:
        model_path = weights_path

    # Load weights
    if __name__ == '__main__':
        logging.info("Loading weights from %s", model_path)

    if weights_path.lower() == "coco":
        # Exclude the last layers because they require a matching number of classes
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)

    return model

def zip_results(result):
    new_results =[]
    for i in range(0,len(result["class_ids"])):
        res = {"class_id": result["class_ids"][i], "mask": result["masks"][i], "roi": result["rois"][i], "score": result["scores"][i]}
        #ROI - y1, x1, y2, x2
        new_results.append(res)
    return new_results

############################################################
# Execution
############################################################
if __name__ == '__main__':
    import argparse
    import logging
    
    def GetParser():
        parser = argparse.ArgumentParser(
            description='Train Mask R-CNN on custom dataset.')
        parser.add_argument("mode",
                            metavar="<command>",
                            help="'train' or 'server'")
        parser.add_argument('-D', '--dataset', required=False,
                            metavar="/path/to/annotations/filename",
                            help='Path to VIA annotation file. Must be on the same directory as images')
        parser.add_argument('-V', '--val', required=False,
                            default=0.2, type=float,
                            help='Percentage of dataset to use for evaluation (default=0.2)')
        parser.add_argument('-m', '--model', required=False,
                            metavar="/path/to/weights.h5",
                            help="Path to weights .h5 file, or 'coco', or 'imagenet' (default), or 'last'",
                            default='imagenet')
        parser.add_argument('-c', '--config', required=False,
                            metavar="/path/to/config.json",
                            help="Path to configuration .json file",
                            default=None)
        parser.add_argument('--logs', required=False,
                            default=DEFAULT_LOGS_DIR,
                            metavar="/path/to/logs/",
                            help='Logs and checkpoints directory (default=logs/)')
        parser.add_argument('--docker',
                            help='Use this if running on the \'pedrosc/mask-rcnn\' docker to use the pre-downloaded models',
                            action="store_true")
        parser.add_argument('-p','--port', required=False, type=int,
                            help='Port to listen to when in server mode')
        
        group = parser.add_mutually_exclusive_group()
        group.add_argument('-d', '--debug', required=False,
                            help="Print lots of debugging statements",
                            action="store_const", dest="loglevel",
                            const=logging.DEBUG)
        group.add_argument('-v', '--verbose', required=False,
                            help="Be verbose",
                            action="store_const", dest="loglevel", const=logging.INFO)
        parser.set_defaults(loglevel=logging.WARNING)
        return parser


    def LoadDatasets(train_data, val_data): # TODO: Remover
        # Training dataset
        logging.info("Loading Training Dataset")
        dataset_train = MastersDataset()
        dataset_train.load_masters(train_data)
        dataset_train.prepare()

        # Validation dataset
        logging.info("Loading Validation Dataset")
        dataset_val = MastersDataset()
        dataset_val.load_masters(val_data)
        dataset_val.prepare()
        
        return dataset_train, dataset_val
        
    def TrainModel(model, config, dataset_train, dataset_val):
        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # TODO: Ler cronograma de treinamento de um arquivo? Ou colocar em Config?
        # Training - Stage 1
        logging.info("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        logging.info("Fine tuning Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        logging.info("Fine tuning all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

#####################
# Start of programm
#####################

    # Parse command line arguments
    args = GetParser().parse_args()
    logging.basicConfig(level=args.loglevel, format='%(asctime)s - %(levelname)s: %(message)s')

    if args.mode == 'train':
        assert args.dataset
    if args.mode == 'server':
        assert args.port

    logging.info("Mode: %s", args.mode)
    logging.info("Model: %s", args.model)
    if args.mode == 'train':
        logging.info("Dataset: %s", args.dataset)
        logging.info("Eval Percent: %s", args.val)
    if args.mode == 'server':
        logging.info("Listening to Port: %s", args.port)
    logging.info("Logs: %s", args.logs)
    
    # Configurations
    logging.debug("Creating Config")
    config = GetConfig(args.mode, args.config)
    logging.debug("Config created")
    #config.display()
    
    # Create model
    logging.debug("Creating Model")
    model = GetModel(args.mode, args.model, args.logs, config, args.docker)
    logging.debug("Model created")
    
    if args.mode == "train":
        (dataset_train, dataset_val) = MastersDataset.auto_split_validation(args.dataset, args.val)
        TrainModel(model, config, dataset_train, dataset_val)
    elif args.mode == "server":
        import socket, pickle
        from socketUtils import *

        HOST = socket.gethostbyname(socket.gethostname()) # Host IP
        PORT = args.port  # 50007 Arbitrary non-privileged port
        logging.info('Host:Port - %s:%s', HOST, PORT)

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))

        logging.info('Bound and Listening...')
        s.listen(1)
        while True:
            conn, addr = s.accept()
            logging.info('Connected by %s', addr)

            while True:
                try:
                    data = recv_msg(conn)
                    if not data: break
                    eval_data = pickle.loads(data)
                    logging.debug('Received Package: %s', eval_data.img_id)

                    logging.info('Running Detection...')
                    results = model.detect([eval_data.data])
                    results = zip_results(results[0])

                    logging.info('Building Response...')
                    response = EvaluationData(eval_data.img_id, results)
                    data = pickle.dumps(response)
                    
                    logging.info('Sending Results')
                    send_msg(conn, data)
                except Exception as e:
                    print(e)
                    break
            conn.close()

    else:
        logging.error("'%s' is not recognized. Use 'train' or 'server'", args.mode)
    
    
    
    
    
    
