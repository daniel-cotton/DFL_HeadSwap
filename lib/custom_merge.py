def merge(base_img_path, seg_image_path):
    from PIL import Image

    background = Image.open('./' + base_img_path)
    foreground = Image.open('./' + seg_image_path)

    background.paste(foreground, (0, 0), foreground)

    return background
