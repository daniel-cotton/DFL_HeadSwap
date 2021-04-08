import traceback
import math
import multiprocessing
import operator
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as npla

import facelib
from core import imagelib
from core import mathlib
from facelib import FaceType, LandmarksProcessor
from core.interact import interact as io
from core.joblib import Subprocessor
from core.leras import nn
from core import pathex
from core.cv2ex import *
from DFLIMG import *
from mainscripts import Extractor

DEBUG = False

def extract(detector=None,
         input_path=None,
         output_path=None,
         output_debug=None,
         manual_fix=False,
         manual_output_debug_fix=False,
         manual_window_size=1368,
         face_type='head',
         max_faces_from_image=None,
         image_size=None,
         jpeg_quality=None,
         cpu_only = False,
         force_gpu_idxs = None,
         ):

    if not input_path.exists():
        io.log_err ('Input directory not found. Please ensure it exists.')
        return

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if face_type is not None:
        face_type = FaceType.fromString(face_type)

    input_image_paths = pathex.get_image_unique_filestem_paths(input_path, verbose_print_func=io.log_info)
    output_images_paths = pathex.get_image_paths(output_path)
    output_debug_path = output_path.parent / (output_path.name + '_debug')

    continue_extraction = True

    device_config = nn.DeviceConfig.GPUIndexes( force_gpu_idxs or nn.ask_choose_device_idxs(choose_only_one=detector=='manual', suggest_all_gpu=True) ) \
                    if not cpu_only else nn.DeviceConfig.CPU()

    if image_size is None:
        image_size = 768

    if jpeg_quality is None:
        jpeg_quality = 90

    if detector is None:
        detector = 's3fd'


    if output_debug is None:
        output_debug = False

    if output_debug:
        output_debug_path.mkdir(parents=True, exist_ok=True)

    if manual_output_debug_fix:
        if not output_debug_path.exists():
            io.log_err(f'{output_debug_path} not found. Re-extract faces with "Write debug images" option.')
            return
        else:
            detector = 'manual'
            io.log_info('Performing re-extract frames which were deleted from _debug directory.')

            input_image_paths = Extractor.DeletedFilesSearcherSubprocessor (input_image_paths, pathex.get_image_paths(output_debug_path) ).run()
            input_image_paths = sorted (input_image_paths)
            io.log_info('Found %d images.' % (len(input_image_paths)))
    else:
        if not continue_extraction and output_debug_path.exists():
            for filename in pathex.get_image_paths(output_debug_path):
                Path(filename).unlink()

    images_found = len(input_image_paths)
    faces_detected = 0
    if images_found != 0:
        if detector == 'manual':
            io.log_info ('Performing manual extract...')
            data = Extractor.ExtractSubprocessor ([ Extractor.ExtractSubprocessor.Data(Path(filename)) for filename in input_image_paths ], 'landmarks-manual', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, manual_window_size=manual_window_size, device_config=device_config).run()

            io.log_info ('Performing 3rd pass...')
            data = Extractor.ExtractSubprocessor (data, 'final', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, final_output_path=output_path, device_config=device_config).run()

        else:
            io.log_info ('Extracting faces...')
            data = Extractor.ExtractSubprocessor ([ Extractor.ExtractSubprocessor.Data(Path(filename)) for filename in input_image_paths ],
                                         'all',
                                         image_size,
                                         jpeg_quality,
                                         face_type,
                                         output_debug_path if output_debug else None,
                                         max_faces_from_image=max_faces_from_image,
                                         final_output_path=output_path,
                                         device_config=device_config).run()

        faces_detected += sum([d.faces_detected for d in data])

        if manual_fix:
            if all ( np.array ( [ d.faces_detected > 0 for d in data] ) == True ):
                io.log_info ('All faces are detected, manual fix not needed.')
            else:
                fix_data = [ Extractor.ExtractSubprocessor.Data(d.filepath) for d in data if d.faces_detected == 0 ]
                io.log_info ('Performing manual fix for %d images...' % (len(fix_data)) )
                fix_data = Extractor.ExtractSubprocessor (fix_data, 'landmarks-manual', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, manual_window_size=manual_window_size, device_config=device_config).run()
                fix_data = Extractor.ExtractSubprocessor (fix_data, 'final', image_size, jpeg_quality, face_type, output_debug_path if output_debug else None, final_output_path=output_path, device_config=device_config).run()
                faces_detected += sum([d.faces_detected for d in fix_data])


    io.log_info ('-------------------------')
    io.log_info ('Images found:        %d' % (images_found) )
    io.log_info ('Faces detected:      %d' % (faces_detected) )
    io.log_info ('-------------------------')
