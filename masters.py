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
#import time
import numpy as np
import imgaug
import json
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from PIL import Image, ImageDraw


MODEL_NAME = "masters" # TODO: Model name
CLASSES = ["Telefone", "Caneca", "Controle Remoto", "Garrafa", "Mão"]

ROOT_DIR = os.getcwd() # Root directory of the project
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs") # Default directory to save logs and model checkpoints

############################################################
#  Configurations
############################################################

class MastersConfig(Config):
    """Configuration for training on custom Dataset.
    Derives from the base Config class and overrides values specific dataset."""
    
    # Give the configuration a recognizable name
    NAME = MODEL_NAME

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    #IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CLASSES)

############################################################
#  Dataset
############################################################

class MastersDataset(utils.Dataset):
    # New loading method.
    def load_masters(self, annotations_filepath):
        dataset_dir = os.path.dirname(annotations_filepath)
        
        # Add classes
        for i in range(len(CLASSES)):
            self.add_class(MODEL_NAME, i + 1, CLASSES[i])
        
        # Add images
        annotations = json.load(open(annotations_filepath))
        i = 0
        for a in annotations.values():
            self.add_image(
                MODEL_NAME, image_id=i, #TODO: id diferente?
                path=os.path.join(dataset_dir, a['filename']),
                width=a['file_attributes']['Width'],
                height=a['file_attributes']['Height'],
                annotations=a)
            i += 1
    
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
    def polygon_to_mask(self, polygon, width, height):
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
# Execution
############################################################
if __name__ == '__main__':
    import argparse
    import logging
    
    def getParser():
        parser = argparse.ArgumentParser(
            description='Train Mask R-CNN on custom dataset.')
        parser.add_argument("mode",
                            metavar="<command>",
                            help="'train' or 'evaluate'")
        parser.add_argument('-D', '--dataset', required=True,
                            metavar="/path/to/annotations/filename",
                            help='Path to VIA annotation file. Must be on the same directory as images')
        parser.add_argument('-m', '--model', required=True,
                            metavar="/path/to/weights.h5",
                            help="Path to weights .h5 file, or 'coco', or 'imagenet', or 'last'")
        parser.add_argument('--logs', required=False,
                            default=DEFAULT_LOGS_DIR,
                            metavar="/path/to/logs/",
                            help='Logs and checkpoints directory (default=logs/)')
        parser.add_argument('-l', '--limit', required=False,
                            default=500, type=int,
                            metavar="<image count>",
                            help='Images to use for evaluation (default=500)')
        parser.add_argument('--docker',
                            help='Use this if running on the \'pedrosc/mask-rcnn\' docker to use the pre-downloaded models',
                            action="store_true")
        
        group = parser.add_mutually_exclusive_group()
        group.add_argument('-d', '--debug', required=False,
                            help="Print lots of debugging statements",
                            action="store_const", dest="loglevel",
                            const=logging.DEBUG)#, default=logging.WARNING)
        group.add_argument('-v', '--verbose', required=False,
                            help="Be verbose",
                            action="store_const", dest="loglevel", const=logging.INFO)
        parser.set_defaults(loglevel=logging.WARNING)
        return parser
    
    def GetConfig(mode):
        if mode == "train":
            return MastersConfig()
        else:
            class InferenceConfig(MastersConfig):
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1
                DETECTION_MIN_CONFIDENCE = 0
            return InferenceConfig()
        
    def GetModel(args):
        if args.mode == "train":
            model = modellib.MaskRCNN(mode="training", config=config,
                                      model_dir=args.logs)
        else:
            model = modellib.MaskRCNN(mode="inference", config=config,
                                      model_dir=args.logs)
            
        # Select weights file to load
        if args.model.lower() == "last": # Find last trained weights
            model_path = model.find_last()[1]
        elif(args.docker):
            import dockerUtils
            model_path = dockerUtils.GetModelPath(args.model)
        else:
            #if args.model.lower() == "coco":
            #    model_path = COCO_MODEL_PATH
            if args.model.lower() == "imagenet":
                # Start from ImageNet trained weights
                model_path = model.get_imagenet_weights()
            else:
                model_path = args.model
    
        # Load weights
        logging.info("Loading weights from %s", model_path)
        model.load_weights(model_path, by_name=True)
        
        return model

    def LoadDatasets(train_data, val_data):
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
    args = getParser().parse_args()
    logging.basicConfig(level=args.loglevel, format='%(asctime)s - %(message)s')
    
    logging.info("Mode: %s", args.mode)
    logging.info("Model: %s", args.model)
    logging.info("Dataset: %s", args.dataset)
    logging.info("Logs: %s", args.logs)
    logging.info("Eval Limit: %s", args.limit)
    
    # Configurations
    logging.debug("Creating Config")
    config = GetConfig(args.mode)
    logging.debug("Config created")
    #config.display()
    
    # Create model
    logging.debug("Creating Model")
    model = GetModel(args)
    logging.debug("Model created")
    
    if(args.mode == "train"):
        (dataset_train, dataset_val) = LoadDatasets(args.dataset, args.dataset)
        TrainModel(model, config, dataset_train, dataset_val)
    elif(args.mode == "evaluate"):
        #TODO
        logging.warning("Not implemented yet")
    else:
        logging.error("'%s' is not recognized. Use 'train' or 'evaluate'", args.mode)
    
    
    
    
    
    
