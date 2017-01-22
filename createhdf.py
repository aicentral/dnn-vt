import h5py,sys
from PIL import Image
caffe_root='/home/ahmed/caffe/'
sys.path.append('/home/ahmed/caffe/python')
import caffe
import numpy as np
datadir='/media/ahmed/OS/text_detection/coco/txt/'
SIZE = 32 # fixed size to all images
chunksize=10000
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
transformer = caffe.io.Transformer({'data': (1,3,SIZE,SIZE)})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

transformer_label = caffe.io.Transformer({'data': (1,1,SIZE,SIZE)})
transformer_label.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer_label.set_mean('data', np.asarray([mu.mean()]) )           # subtract the dataset-mean value in each channel
def loadlabel(dataDir0,cnt):
        im = Image.open('%s/%s'%(dataDir0, '%d.png'%(cnt) ))
        (width, height) = im.size
        img0 = list(im.getdata())
        img0 = np.array(img0)
        img0 = img0.reshape((height, width))
        return np.expand_dims(img0, axis=2)
        #im = im.convert('L')
for phase in ['train','test']:
    with open( '/home/ahmed/txt/%s_reduced.txt'%(phase if phase=='train' else 'val'), 'r' ) as T :
        alllines = T.readlines()
 
    L= open('/media/ahmed/OS/text_detection/coco/txt/h5/%s_list.txt'%(phase),'w')
    L2= open('/media/ahmed/OS/text_detection/coco/txt/h5/%s_list_local.txt'%(phase),'w')
    #L2= open('/media/ahmed/OS/text_detection/coco/cocoswt/h5/test_list.txt','w')
    cnt=0
    totalcnt=len(alllines)
    for j in range(0,totalcnt,chunksize):
    #for j in range(0,1):
        lines=alllines[j:(j+chunksize)]
        H=h5py.File('/media/ahmed/OS/text_detection/coco/txt/h5/%s/%s_%s.h5'%(phase,phase,j),'w')
        H.create_dataset('img', (len(lines), 3,SIZE,SIZE), dtype=np.uint8)
        H.create_dataset('labels', (len(lines), 2), dtype=np.uint8)
        lines=alllines[j:min(len(alllines),(j+chunksize))]
        for i,l in enumerate(lines):
            if (not i % 100):
                print '%s, chunk %d K, %d of %d'%(phase,j/1000,i+j,totalcnt)
            l=l.split(' ')
            img = caffe.io.load_image(datadir+l[0]) 
            #img0 = caffe.io.load_image('%s/%s'%(dataDir0, '%d.png'%(cnt) ))
            cnt+=1
            img = np.array(transformer.preprocess('data', img),dtype='int')
            lbl=np.array([int(l[1][0]) , 1-int(l[1][0])],dtype='int')
            # you may apply other input transformations here...
            #break
            H['img'][i] = img
            H['labels'][i] = lbl
            #H['label'][i] = float(sp[1])
        #with h5py.File('txt36train.h5','w') as H:
        #    H.create_dataset( 'X', data=X ) # note the name X given to the dataset!
        #    H.create_dataset( 'y', data=y ) # note the name y given to the dataset!
        print >>L,'/work/newriver/nady/data/txt/h5/%s/%s_%s.h5'%(phase,phase,j) 
        print >>L2,'/media/ahmed/OS/text_detection/coco/txt/h5/%s/%s_%s.h5'%(phase,phase,j) 
        #print >>L,'/home/ahmed/coco/mini/h5train/txt36train_%s.h5'%(j)
        #L.write('/home/ahmed/coco/h5train/txt36train_%s.h5'%(j) ) # list all h5 files you are going to use
        h5py.File.close(H)
    #with open('/home/ahmed/coco/txt100va_list.txt','w') as L:
       # L.write( '/home/nady/coco/txt100val.h5' ) # list all h5 files you are going to use
    L.close();
    L2.close();
