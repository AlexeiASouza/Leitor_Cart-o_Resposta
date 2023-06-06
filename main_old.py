####################LIBS###########################################
import pytesseract #OCR

import os
import argparse #Argumentos
#import math
#import random
#import imutils
from collections import deque
from functions import *
from global_variables import *

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2
import numpy as np
pytesseract.pytesseract.tesseract_cmd = tesseractOCR_path

from pdf2image import convert_from_path, convert_from_bytes
############################################################################


DETECTION_MODEL_NAME = detection_model_name

paths = {'CHECKPOINT_PATH': os.path.join('MODELS',DETECTION_MODEL_NAME)
}

files = {'PIPELINE_CONFIG' : os.path.join('MODELS',DETECTION_MODEL_NAME, 'pipeline.config'),
         'LABELMAP' : os.path.join('label_map.pbtxt')
         
}


#Detection Model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-4')).expect_partial()

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image")
parser.add_argument("-d","--dest")
args = parser.parse_args()

Images = convert_from_path(args.image)
np_images = list(map(lambda x: np.array(x),Images))
dest_path = args.dest

if __name__ == "__main__":
    for idx,img in enumerate(np_images):
        try:
            marked,inscricao,answers = get_info(img,detection_model)
            str_insc = inscricao.strip()

            answers_dict = get_answers(answers,n_grade_lines,n_grade_cols)

            marked = get_image_with_results(marked,answers_dict)

            cv2.imwrite(os.path.join(dest_path,str_insc + '.jpg'),marked)
            create_file_from_dict(answers_dict,os.path.join(dest_path,str_insc + '.csv'))
    

        except:
            with open('error.txt','a') as f:
                    f.write(str(idx) + '\n')
