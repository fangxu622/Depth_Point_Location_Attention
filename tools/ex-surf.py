import cv2
import os

Sequen=[1,4]
datadir = '/media/fangxu/Disk4T/LQ/data/stairs/'
outputdir = '/media/fangxu/Disk4T/LQ/SURFKeyPoint/stairs'


depthimgs=[]

sift  = cv2.xfeatures2d.SURF_create()

for i in Sequen:
    if i < 10:
        sequenceDir = datadir + "/seq-0{}/".format(i)
        outputdirSeq = outputdir + "/seq-0{}/".format(i)
    else:
        sequenceDir = datadir + "/seq-{}/".format(i)
        outputdirSeq = outputdir + "/seq-{}/".format(i)

    if os.path.exists(outputdirSeq) == False:
        os.makedirs(outputdirSeq)

    poselabelsNames = os.listdir(sequenceDir)

    for j in range(0, len(poselabelsNames)):
        name = poselabelsNames[j]
        if name.endswith(".txt"):

            imgCount = name.split('.')[0]

            path = sequenceDir + imgCount + '.depth.png'
            outpath = outputdirSeq+imgCount+'.sift.txt'
            fp = open(outpath,'w')
            Img = cv2.imread(path,cv2.CV_16UC1)
            Img = (Img / 256).astype('uint8')
            keypoints, descriptors = sift.detectAndCompute(Img, None)
            # keypoints, descriptors = sift.detect(Img,None,useProvidedKeypoints = False)
            for point in keypoints:
                pt = point.pt
                fp.write(str(pt[0]))
                fp.write(' ')
                fp.write(str(pt[1]))
                fp.write('\n')

            fp.close()
            t =1







t =1