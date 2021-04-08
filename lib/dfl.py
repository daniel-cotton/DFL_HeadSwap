import os
import sys
import traceback
import queue
import threading
import time
import numpy as np
import itertools
from pathlib import Path
from core import pathex
from core import imagelib
from core import osex
import cv2
import models
from core.interact import interact as io

def extract(input_dir, output_dir):
    osex.set_process_lowest_prio()
    from lib.components import extractor
    extractor.extract( detector                = 's3fd',
                    input_path              = Path(input_dir),
                    output_path             = Path(output_dir),
                    output_debug            = None,
                    manual_fix              = False,
                    manual_output_debug_fix = False,
                    manual_window_size      = 1368,
                    face_type               = 'head',
                    max_faces_from_image    = None,
                    image_size              = None,
                    jpeg_quality            = None,
                    cpu_only                = True,
                    force_gpu_idxs          = None,
    )

def train(model_type, model_name, model_dir, training_data_src_dir, training_data_dst_dir):
    from lib.components import trainer
    osex.set_process_lowest_prio()


    kwargs = {
        'model_class_name'         : model_type,
        'saved_models_path'        : Path(model_dir),
        'training_data_src_path'   : Path(training_data_src_dir),
        'training_data_dst_path'   : Path(training_data_dst_dir),
        'pretraining_data_path'    : None,
        'pretrained_model_path'    : None,
        'no_preview'               : True,
        'force_model_name'         : False,
        'force_gpu_idxs'           : None,
        'cpu_only'                 : True,
        'silent_start'             : False,
        'execute_programs'         : [],
        'debug'                    : False,
        'force_model_class_name'   : model_name
    }
    io.log_info ("Running trainer.\r\n")

    no_preview = kwargs.get('no_preview', False)

    s2c = queue.Queue()
    c2s = queue.Queue()

    e = threading.Event()
    thread = threading.Thread(target=trainer.trainerThread, args=(s2c, c2s, e), kwargs=kwargs )
    thread.start()

    e.wait() #Wait for inital load to occur.


    while True:
        if not c2s.empty():
            input = c2s.get()
            op = input.get('op','')
            if op == 'close':
                break
        try:
            io.process_messages(0.1)
        except KeyboardInterrupt:
            s2c.put ( {'op': 'close'} )
    print('completed train....')

def merge(model_type, model_name, model_dir, src_aligned, dst_input, output_dir, output_mask_dir):
    osex.set_process_lowest_prio()
    from lib.components import merger
    merger.merge (  model_class_name       = model_type,
                    saved_models_path      = Path(model_dir),
                    force_model_name       = model_name,
                    input_path             = Path(dst_input),
                    output_path            = Path(output_dir),
                    output_mask_path       = Path(output_mask_dir),
                    aligned_path           = Path(src_aligned),
                    force_gpu_idxs         = False,
                    cpu_only               = True)