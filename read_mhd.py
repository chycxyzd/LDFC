import SimpleITK as sitk
import matplotlib.pyplot as plt

case_path = r'lung_nodule_dataset/MHD/1.mhd'  # Source Image Path
itkimage = sitk.ReadImage(case_path)  # This section gives information about the image, which can be printed out and looked at

image = sitk.GetArrayFromImage(itkimage)  # z,y,x

# View X-th image
x = 44
plt.figure(1)
plt.imshow(image[x, :, :])
a = image[x, :, :]
print(a.shape)
plt.show()



