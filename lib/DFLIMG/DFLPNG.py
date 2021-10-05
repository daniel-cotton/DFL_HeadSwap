

import imageio
import binascii
import pickle
import traceback

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *
from core.imagelib import SegIEPolys
from core.interact import interact as io
from core.structex import *
from facelib import FaceType


class DFLPNG(object):

    _END_CHUNK_TYPE = 'IEND'
    _CUSTOM_CHUNK_TYPE = 'jaNf'

    def __init__(self, filename):
        self.filename = filename
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = None
        self.shape = None
        self.img = None

    @staticmethod
    def load_raw(filename, loader_func=None):
        try:
            if loader_func is not None:
                #data = loader_func(filename)
                with open(filename, "rb") as f:
                    data = f.read()
            else:
                with open(filename, "rb") as f:
                    data = f.read()
        except:
            raise FileNotFoundError(filename)

        try:
            inst = DFLPNG(filename)
            inst.data = data
            inst.length = len(data)


            return inst
        except Exception as e:
            raise Exception (f"Corrupted JPG file {filename} {e}")

    @staticmethod
    def load(filename, loader_func=None):
        try:
            inst = DFLPNG.load_raw (filename, loader_func=loader_func)
            inst.dfl_dict = {}

            byte_counter = 0
            byte_counter+=8 #png header

            png_data = bytearray(inst.data[:byte_counter])
            payload = None

            while byte_counter < inst.length:
                chunk_size_b = inst.data[byte_counter:byte_counter+4] # to int 
                chunk_size= int(chunk_size_b.hex() , 16)
                byte_counter+=4

                chunk_type_b = inst.data[byte_counter:byte_counter+4] #to ascii 
                chunk_type = chunk_type_b.decode() 
                byte_counter+=4

                if chunk_type == DFLPNG._END_CHUNK_TYPE:
                    png_data.extend(chunk_size_b)
                    png_data.extend(chunk_type_b)
                    crc_b = inst.data[byte_counter:byte_counter+4]
                    png_data.extend(crc_b)
                    break

                content = inst.data[byte_counter:byte_counter+chunk_size]
                byte_counter+=chunk_size

                crc_b = inst.data[byte_counter:byte_counter+4] # to hex
                crc = crc_b.hex() 
                byte_counter+=4

                if chunk_type == DFLPNG._CUSTOM_CHUNK_TYPE:
                    payload = content
                    pass

                png_data.extend(chunk_size_b)
                png_data.extend(chunk_type_b)
                png_data.extend(content)
                png_data.extend(crc_b)

            if payload is not None:
                inst.dfl_dict = pickle.loads(payload)
            inst.chunks = png_data


            return inst
        except Exception as e:
            io.log_err (f'Exception occured while DFLPNG.load : {traceback.format_exc()}')
            return None

    def has_data(self):
        return len(self.dfl_dict.keys()) != 0

    def save(self):
        try:
            with open(self.filename, "wb") as f:
                f.write ( self.dump() )
        except:
            raise Exception( f'cannot save {self.filename}' )

    def dump(self):


        dict_data = self.dfl_dict

        # Remove None keys
        for key in list(dict_data.keys()):
            if dict_data[key] is None:
                dict_data.pop(key)
        png_data = self.chunks
        payload = pickle.dumps(dict_data)

        png_with_payload = bytearray(png_data[:-12]) # skip IEND chunk
        chunk_size = len(payload)

        # # Create a temporary byte array for the CRC check.
        tmp_bytes = bytearray()

        # # First write the chunk type
        tmp_bytes.extend(bytearray(DFLPNG._CUSTOM_CHUNK_TYPE, "ascii" ))

        # # Now write the bytes of whatever we're trying to hide
        tmp_bytes.extend(payload)

        # Write the chunk size
        png_with_payload.extend(bytearray(struct.pack('!i', chunk_size)))

        # And the type
        png_with_payload.extend(bytearray(DFLPNG._CUSTOM_CHUNK_TYPE, "ascii"))

        png_with_payload.extend(payload)

        crc = binascii.crc32(tmp_bytes) & 0xffffffff
        del tmp_bytes

        #print(len((bytearray(struct.pack('!I', crc)))))
        png_with_payload.extend(bytearray(struct.pack('!I', crc)))

        #tmp_bytes = bytearray()

        # # First write the chunk type
        #tmp_bytes.extend(bytearray(DFLPNG._END_CHUNK_TYPE, "ascii" ))

        # # Now write the bytes of whatever we're trying to hide
        #tmp_bytes.extend(0)
        #crc = binascii.crc32(tmp_bytes) & 0xffffffff
        #del tmp_bytes

        # Write the end chunk. Start with the size.
        #png_with_payload.extend(bytearray(struct.pack('!i', 0)))
        # Then the chunk type.
        #png_with_payload.extend(bytearray(DFLPNG._END_CHUNK_TYPE, "ascii"))
        #png_with_payload.extend(bytearray(struct.pack('!I', crc)))
        png_with_payload.extend(png_data[-12:])

        return png_with_payload


    def get_img(self):
        if self.img is None:
            #img = imageio.imread(self.filename)
            #self.img = np.asarray(img[:,:,::-1])
            self.img = cv2_imread(self.filename)
        return self.img

    def get_shape(self):
        if self.shape is None:
            img = self.get_img()
            if img is not None:
                self.shape = img.shape
        return self.shape

    def get_height(self):
      height, width, channels = self.get_shape()
      return height


    def get_dict(self):
        return self.dfl_dict

    def set_dict (self, dict_data=None):
        self.dfl_dict = dict_data

    def get_face_type(self):            return self.dfl_dict.get('face_type', FaceType.toString (FaceType.FULL) )
    def set_face_type(self, face_type): self.dfl_dict['face_type'] = face_type

    def get_landmarks(self):            return np.array ( self.dfl_dict['landmarks'] )
    def set_landmarks(self, landmarks): self.dfl_dict['landmarks'] = landmarks

    def get_eyebrows_expand_mod(self):                      return self.dfl_dict.get ('eyebrows_expand_mod', 1.0)
    def set_eyebrows_expand_mod(self, eyebrows_expand_mod): self.dfl_dict['eyebrows_expand_mod'] = eyebrows_expand_mod

    def get_source_filename(self):                  return self.dfl_dict.get ('source_filename', None)
    def set_source_filename(self, source_filename): self.dfl_dict['source_filename'] = source_filename

    def get_source_rect(self):              return self.dfl_dict.get ('source_rect', None)
    def set_source_rect(self, source_rect): self.dfl_dict['source_rect'] = source_rect

    def get_source_landmarks(self):                     return np.array ( self.dfl_dict.get('source_landmarks', None) )
    def set_source_landmarks(self, source_landmarks):   self.dfl_dict['source_landmarks'] = source_landmarks

    def get_image_to_face_mat(self):
        mat = self.dfl_dict.get ('image_to_face_mat', None)
        if mat is not None:
            return np.array (mat)
        return None
    def set_image_to_face_mat(self, image_to_face_mat):   self.dfl_dict['image_to_face_mat'] = image_to_face_mat

    def has_seg_ie_polys(self):
        return self.dfl_dict.get('seg_ie_polys',None) is not None

    def get_seg_ie_polys(self):
        d = self.dfl_dict.get('seg_ie_polys',None)
        if d is not None:
            d = SegIEPolys.load(d)
        else:
            d = SegIEPolys()

        return d

    def set_seg_ie_polys(self, seg_ie_polys):
        if seg_ie_polys is not None:
            if not isinstance(seg_ie_polys, SegIEPolys):
                raise ValueError('seg_ie_polys should be instance of SegIEPolys')

            if seg_ie_polys.has_polys():
                seg_ie_polys = seg_ie_polys.dump()
            else:
                seg_ie_polys = None

        self.dfl_dict['seg_ie_polys'] = seg_ie_polys

    def has_xseg_mask(self):
        return self.dfl_dict.get('xseg_mask',None) is not None

    def get_xseg_mask_compressed(self):
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None

        return mask_buf

    def get_xseg_mask(self):
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None

        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = img[...,None]

        return img.astype(np.float32) / 255.0


    def set_xseg_mask(self, mask_a):
        if mask_a is None:
            self.dfl_dict['xseg_mask'] = None
            return

        mask_a = imagelib.normalize_channels(mask_a, 1)
        img_data = np.clip( mask_a*255, 0, 255 ).astype(np.uint8)

        data_max_len = 8192

        ret, buf = cv2.imencode('.png', img_data)

        if not ret or len(buf) > data_max_len:
            for jpeg_quality in range(100,-1,-1):
                ret, buf = cv2.imencode( '.jpg', img_data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] )
                if ret and len(buf) <= data_max_len:
                    break

        if not ret:
            raise Exception("set_xseg_mask: unable to generate image data for set_xseg_mask")

        self.dfl_dict['xseg_mask'] = buf 
