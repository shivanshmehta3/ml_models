import matplotlib.image as image
import matplotlib.pyplot as plt
img = image.imread("E:\my_projects\\trash-segregater\data\images\s1.png")
print(img)
imshw=plt.imshow(img)
print(img.shape)