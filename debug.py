#%%
import sys
import os
sys.path.append(os.getenv('DEEPFACELAB_PATH'))

#%%
# # Fix for linux
import multiprocessing
# if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
from core.leras import nn
nn.initialize_main_env()
import os
import sys
import time
import argparse
import subprocess

from core import pathex
from core import osex
from pathlib import Path
from core.interact import interact as io
from distutils.dir_util import copy_tree

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
    raise Exception("This program requires at least Python 3.6")

print("Hello world")
#%%
# init workspace
subprocess.call(['sh', './init.sh'])
#%%
copy_tree("./src/", os.getenv('DEEPFACELAB_WORKSPACE') + "/data_src/")
copy_tree("./dst/", os.getenv('DEEPFACELAB_WORKSPACE') + "/data_dst/")
copy_tree("./model_data/", os.getenv('DEEPFACELAB_WORKSPACE') + "/model/")

#%%
from lib import dfl

src_aligned = os.getenv('DEEPFACELAB_WORKSPACE') + "/data_src/aligned"
dst_aligned = os.getenv('DEEPFACELAB_WORKSPACE') + "/data_dst/aligned"
dfl.extract(os.getenv('DEEPFACELAB_WORKSPACE') + "/data_src/", src_aligned)
dfl.extract(os.getenv('DEEPFACELAB_WORKSPACE') + "/data_dst/", dst_aligned)
#%%
from lib import dfl

src_aligned = os.getenv('DEEPFACELAB_WORKSPACE') + "/data_src/aligned"
dst_aligned = os.getenv('DEEPFACELAB_WORKSPACE') + "/data_dst/aligned"
model_dir = os.getenv('DEEPFACELAB_WORKSPACE') + "/model"

dfl.train('SAEHD', 'head-masked-2_SAEHD', model_dir, src_aligned, dst_aligned)
# %%
from lib import dfl

src_aligned = os.getenv('DEEPFACELAB_WORKSPACE') + "/data_dst/aligned"
dst_aligned = os.getenv('DEEPFACELAB_WORKSPACE') + "/data_dst"
out_merged = os.getenv('DEEPFACELAB_WORKSPACE') + "/merged"
out_mask = os.getenv('DEEPFACELAB_WORKSPACE') + "/merged_mask"
model_dir = os.getenv('DEEPFACELAB_WORKSPACE') + "/model"

dfl.merge('SAEHD', 'head-masked-2_SAEHD', model_dir, src_aligned, dst_aligned, out_merged, out_mask)
# %%

import pymatting_aot.cc

# %%
from lib import segmentation
from lib import custom_merge
from PIL import Image
import cv2

dst_path = './dst/11833.png'
demo_path = './11833.png'
tmp = './11833-tmp.png'
demo_out = './11833-out.png'
demo_out_seg = './11833-seg.png'


cvimg = cv2.imread(demo_path)
cvout = segmentation.remove_background(cvimg, threshold=250.)
cv2.imwrite(demo_out, cvout)


segmented = segmentation.get_image_segmentation(demo_out)
segmented.save(demo_out_seg)

# merged = custom_merge.merge(dst_path, demo_out)
# merged.save(demo_out)

# %%
import numpy as np
import cv2 as cv
img = cv.imread('11840.png',0)
ret,thresh = cv.threshold(img,127,255,0)
contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv.moments(cnt)
print( M )
cv.drawContours(img, contours, -1, (0,255,0), 3)
cv.imwrite('11840-cv.png', img)
# %%
import numpy as np
import cv2
image = cv2.imread('11840.png')

b = image.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0

g = image.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = image.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0

cv.imwrite('11840-r.png', r)
cv.imwrite('11840-g.png', g)
cv.imwrite('11840-b.png', b)


img = cv.imread('11840-b.png',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imwrite('11840-cv.png', img)
# %%
import cv2
import numpy as np

input = os.getenv('DEEPFACELAB_WORKSPACE') + "/merged/11840.png"
input_mask = os.getenv('DEEPFACELAB_WORKSPACE') + "/merged_mask/11840.png"


img = cv2.imread(input)
img_mask = cv2.imread(input_mask,0)
dilation = cv2.dilate(img,img_mask,iterations = 1)

# %%
import cv2
import numpy as np

input = os.getenv('DEEPFACELAB_WORKSPACE') + "/merged/11840.png"
input_mask = os.getenv('DEEPFACELAB_WORKSPACE') + "/merged_mask/11840.png"
bg_path = os.getenv('DEEPFACELAB_WORKSPACE') + "/data_dst/11840.png"

mask = cv2.imread(input_mask)

def fetch_largest_bounding_box_from_image_contours(mask):
    ###binarising
    gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    ###applying morphological operations to dilate the image
    kernel=np.ones((10,10),np.uint8)
    dilated=cv2.dilate(th2,kernel,iterations=3)

    contours,hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ### converting to bounding boxes from polygon
    contours=[cv2.boundingRect(cnt) for cnt in contours]

    largestContour = None

    ### drawing rectangle for each contour for visualising
    for cnt in contours:
        x,y,w,h=cnt
        area = w * h
        if x == 0:
            # do nothing
            x = 0
        else:
            if largestContour == None:
                largestContour = cnt
            else:
                x,y,w,h = largestContour
                existingArea = w * h

                if existingArea < area:
                    largestContour = cnt
            # cv2.rectangle(mask,(x,y),(x+w,y+h),(0,255,0),2)
    return largestContour

def convert_to_four_channel(img):
    alpha_channel = np.ones(img.shape, dtype=img.dtype) * 255
    b, g, r = cv2.split(img)
    rgba = [b,g,r, alpha_channel]
    dst = cv2.merge(rgba, 4)
    return rgba

def remove_black_bg_with_threshold(img):
    height, width, channels = img.shape
    img = cv2.medianBlur(img,5)
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,40,255,cv2.THRESH_BINARY)
    # alpha = cv2.adaptiveThreshold(tmp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv2.THRESH_BINARY,11,2)
    if channels == 3:
        b, g, r = cv2.split(img)
        rgba = [b,g,r, alpha]
        dst = cv2.merge(rgba,4)
        return dst

    b, g, r, a  = cv2.split(img)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    return dst

def mask_img_rect(img, topLeft, topRight):
    height, width, channels = img.shape
    mask = np.ones(img.shape, dtype="uint8")
    cv2.rectangle(mask, topLeft, topRight, (255,255,255,255), -1)

    res = img * (mask / 255)
    return res

img = cv2.imread(input)
height, width, channels = img.shape

x,y,w,h = fetch_largest_bounding_box_from_image_contours(mask)
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# print(x,y,w,h)
faceMask = (x,y,w,h)

im_bw = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
im_bw = cv2.medianBlur(im_bw,5)

# # faceMaskOverlay = fourImg * im_bw
# faceMaskOverlay = cv2.bitwise_and(img, im_bw)

transparent = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
transparent[:,:,0:3] = img
transparent[:, :, 3] = im_bw

cv2.imwrite('11840-m.png', transparent)

black = (0,0,0)

maxLeft = round(x - (w * 0.1))
minRight = round(x + (w * 1.1))

cv2.rectangle(img, (0,0), (maxLeft,height), black,-1)
cv2.rectangle(img, (minRight,0), (width,height), black,-1)

img = remove_black_bg_with_threshold(img)

x,y,w,h = fetch_largest_bounding_box_from_image_contours(img)

# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.rectangle(img, (0, y + h), (width,height), black,-1)

img = remove_black_bg_with_threshold(img)

cv2.imwrite('11840.png', img)

bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)

mergePoint = round((y + h + faceMask[1] + faceMask[3]) / 2)


bg = mask_img_rect(bg, (0, y + h), (width, height))

cv2.imwrite('11840-bg.png', bg)

from PIL import Image

img = Image.open("11840.png")
overlay = Image.open("11840-m.png")

background = Image.open('11840-bg.png')

background.paste(overlay, (0, 0), overlay)
background.paste(img, (0, 0), img)
background.save('11840.png',"PNG")



# %%
