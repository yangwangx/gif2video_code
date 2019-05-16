import os
download_root = 'http://vision.cs.stonybrook.edu/~yangwang/public/gif2video/pretrained/'
with open('filelist.txt', 'r') as f:
    lines = f.getlines()
    for line in lines:
        model = line.rstrip()
        cmd = 'wget {}{}'.format(download_root, model)
        print('Downloading [' + model +']')
        os.system(cmd)
