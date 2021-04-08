# def get_image_segmentation(image):
#     from fastseg import MobileV3Small
#     model = MobileV3Small.from_pretrained()
#     model.eval()

#     labels = model.predict_one(image)
#     from fastseg.image import colorize, blend

#     print(labels)

#     colorized = colorize(labels) # returns a PIL Image
#     composited = blend(image, colorized) # returns a PIL Image

#     return composited


def get_image_segmentation(image_path):
    import numpy as np
    from rembg.bg import remove
    import io
    from PIL import Image

    f = np.fromfile(image_path)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGBA")

    return img

def remove_background(img, threshold):
    import cv2
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)


    return dst

# def get_image_segmentation(image):
#     import cv2
#     import numpy as np
#     img = cv2.imread(image)
#     mask = np.zeros(img.shape[:2],np.uint8)
#     bgdModel = np.zeros((1,65),np.float64)
#     fgdModel = np.zeros((1,65),np.float64)
#     rect = (50,50,img.shape[1] - 50,img.shape[0] - 50)

#     print(rect)
#     cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
#     mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#     img_cut = img*mask2[:,:,np.newaxis]

#     return img_cut


