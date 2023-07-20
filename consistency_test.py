from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import albumentations as A
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from natsort import natsorted

class ImageTransform:
    def __init__(self,):
        self.transform = A.Compose([
            # A.Resize(height=img_size[0], width=img_size[1]),
            A.Crop(x_min=103, x_max=130, y_min=100, y_max=130),
        ])
    def __call__(self, image):
        return self.transform(image=np.array(image))["image"]

DIR = 'C:/Users/max.polak/PycharmProjects/tactile-sensing/master_students/max_tactile_xai/polycamExplain/explanations/ce-images/'
files = natsorted(os.listdir(DIR))
print(files)
image_files = [os.path.join(DIR, file) for file in files if file.endswith('.png')]

# Apply center crop
transform = ImageTransform()

similarity_scores = []
ref_image = transform(Image.open(image_files[0]))
for file in image_files:
    image = transform(np.array(Image.open(file)))
    plt.imshow(image)
    plt.show()
    similarity_score = np.mean(ssim(ref_image, image))
    similarity_scores.append(similarity_score)

# for i in range(len(image_files)):
#     ref_image = np.array(Image.open(image_files[i]))
#
#     for j in range(i + 1, len(image_files)):
#         image = np.array(Image.open(image_files[j]))
#         similarity_score = abs(msssim(ref_image, image))
#         similarity_scores.append(similarity_score)

print(similarity_scores)
mean_similarity = np.array(similarity_scores).mean()
print(mean_similarity)


# image0 = np.array(Image.open(image_files[1]))
# image1 = np.array(Image.open(image_files[3]))
#
# print("MSE: ", mse(image1,image0))
# print("RMSE: ", rmse(image1, image0))
# print("PSNR: ", psnr(image1, image0))
# print("SSIM: ", ssim(image1, image0))
# print("UQI: ", uqi(image1, image0))
# print("MSSSIM: ", abs(msssim(image1, image0)))
# print("ERGAS: ", ergas(image1, image0))
# print("SCC: ", scc(image1, image0))
# print("RASE: ", rase(image1, image0))
# print("SAM: ", sam(image1, image0))
# print("VIF: ", vifp(image1, image0))