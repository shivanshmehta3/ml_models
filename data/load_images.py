import scipy.io as io
from pathlib import Path
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import data.pca_model.model as pca
import data.config as cfg
from PIL import Image

def loadImages():
    base_path = Path(__file__).parent
    file_path = (base_path / '.\images\\i1.png').resolve()
    print(file_path)
    im = Image.open("E:\my_projects\waste_segregator\data\images\i0.png")
    im = im.resize((cfg.IMAGE_WIDTH,cfg.IMAGE_HEIGHT), Image.ANTIALIAS)
    img = np.array(im)
    print(img.shape)
    img = img[:,:,0:3]
    img_rs = np.resize(img, (1, img.size))

    iter_arr = np.linspace(1,cfg.SIZE_OF_DATASET-1,cfg.SIZE_OF_DATASET-1, dtype=int)
    for iter in iter_arr:
        im = Image.open("E:\my_projects\waste_segregator\data\images\i{}.png".format(iter))
        im = im.resize((cfg.IMAGE_WIDTH,cfg.IMAGE_HEIGHT), Image.ANTIALIAS)
        img = np.array(im)
        img = img[:,:,0:3]
        img1_rs = np.resize(img, (1, img.size))
        img_rs = np.concatenate((img_rs,img1_rs))
    images_data = img_rs
    # img = image.imread("E:\my_projects\waste_segregator\data\images\i1.png")
    # img = img[:,:,0:3]
    # img2 = image.imread("E:\my_projects\waste_segregator\data\images\i2.png")
    # img2 = img2[:,:,0:3]
    # img1_rs = np.resize(img, (1, img.size))
    # img2_rs = np.resize(img2, (1, img2.size))
    # img_rs = np.concatenate((img1_rs,img2_rs))
    # pca_mdl = pca.Model(img_rs)
    # img_rd = pca_mdl.reduce_dimension()