import cv2

def draw(img,left,right,color):
    #img=cv2.imread(spath)
    img=cv2.rectangle(img,left,right,color,3)
    return img

source_path='175.bmp'  # Source Image Path
target_path='175_box.bmp'  # Image Save Path
img=cv2.imread(source_path)
img=draw(img,(145,149),(171,123),(0,255,0)) # (img, (xmin,ymin),(xmax,ymax),(0,255,0))
cv2.imwrite(target_path,img)
