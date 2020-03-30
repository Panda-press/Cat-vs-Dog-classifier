import PIL
from PIL import Image

new_width = 100
new_height = 100

new_image_num = 0

print("---Begining dogs---")

for image_num in range(0,12500):
    try:
        img = Image.open("D:\Dataset\Kaggle_cats_dogs\PetImages\Dog\\"+str(image_num)+".jpg")
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.convert("RGB").save("D:\Dataset\Kaggle_cats_dogs\Processed_Images\Dog\\"+str(new_image_num)+".jpg")
        new_image_num +=1

    except:
        print(image_num)



new_image_num = 0

print("---Begining cats---")
        
for image_num in range(0,12500):
    try:
        img = Image.open("D:\Dataset\Kaggle_cats_dogs\PetImages\Cat\\"+str(image_num)+".jpg")
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.convert("RGB").save("D:\Dataset\Kaggle_cats_dogs\Processed_Images\Cat\\"+str(new_image_num)+".jpg")
        new_image_num +=1

    except:
        print(image_num)