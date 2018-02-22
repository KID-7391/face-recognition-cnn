import os
from PIL import Image


#dir = os.listdir('mydata_before')
#for subject in dir:
#    in_path = 'mydata_before' + '/' +subject
#    out_path = 'mydata' + '/' + subject
#    ls = os.listdir(in_path)
#    for i in ls:
#        img = Image.open(in_path + '/' + i).convert('L').resize((64, 64))
#        img.save(out_path + '/' + i)

dir = os.listdir('mydata')
for subject in dir:
    ls = os.listdir('mydata/' + subject)
    cnt = 31
    for i in ls:
        img = Image.open('mydata/' + subject + '/' + i).transpose(Image.FLIP_LEFT_RIGHT)
        img.save('mydata/' + subject + '/' + i.split(' ')[0] + ' (' + str(cnt) + ').bmp')
        cnt += 1



