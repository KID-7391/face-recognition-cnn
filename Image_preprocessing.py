import os
from PIL import Image


dir = os.listdir('mydata_before')
for subject in dir:
    in_path = 'mydata_before' + '/' +subject
    out_path = 'mydata' + '/' + subject
    ls = os.listdir(in_path)
    for i in ls:
        img = Image.open(in_path + '/' + i).convert('L').resize((64, 64))
        img.save(out_path + '/' + i)
