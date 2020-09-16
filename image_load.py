import tensorflow as tf
import os
import numpy as np

names = ['A30P','A53E','A53T','E46K','G51D','H50Q','WT']
temporal_tag = ['pics2h','pics4h','pics6h','pics16h']

def read_3channel_images(target_path, image_size=256):
    images = []
    labels = []
    for i in range(len(names)):
        files = os.walk(target_path+names[i]+'/')
        files = list(files)[0][2]
        for f in files:
            im = tf.io.read_file(target_path+names[i]+'/'+f)
            im = tf.image.decode_jpeg(im, channels=3)
            im = tf.cast(im, tf.float32)
            im = im / tf.reduce_max(im)
            im = tf.image.resize(im, [image_size, image_size])
            images.append(im.numpy())
            labels.append(i)
    return images,labels

def read_12channel_images(target_path, image_size=256):
    images = []
    labels = []
    for i in range(len(names)):
        files = list(os.walk(target_path))
        imageFiles = list(os.walk(target_path+temporal_tag[0]+'/'))
        for f in imageFiles[1][2]:
            im_mat = []
            try:
                for j in temporal_tag:
                    path = target_path+j+'/'+ names[i] + '/' + f
                    im = tf.io.read_file(path)
                    im = tf.image.decode_jpeg(im, channels=3)
                    im = tf.cast(im, tf.float32)
                    im = im/tf.reduce_max(im)
                    im = tf.image.resize(im, [image_size, image_size])
                    im_mat.append(im.numpy())
                images.append(np.concatenate(im_mat,axis=2))
                labels.append(i)
            except:
                print(f'invalid file name: {f}')
    return images,labels

class KFoldContainer(object):
    def __init__(self,X,y,K):
        self.X = X
        self.y = np.array(y)
        self.K = K
        print(f'number of sample: {self.n_class_sample.__str__():s}')
        self.indices = np.zeros_like(self.y)
        for i,n_sample in enumerate(self.n_class_sample):
            image_pos = np.where(self.y == self.unique_y[i])
            rand_image_pos = np.random.choice(image_pos[0],n_sample,replace=False)
            kfold_index = np.clip(np.array(range(n_sample))*self.K//n_sample,a_min=0,a_max=self.K-1)
            for j,pos in enumerate(rand_image_pos):
                self.indices[pos] = kfold_index[j]

    def get_fold_k(self,k):
        assert k>=0 and k<self.K
        Xtrain = [ x for i,x in enumerate(self.X) if self.indices[i] != k ]
        ytrain = self.y[self.indices != k]
        Xtest = [x for i, x in enumerate(self.X) if self.indices[i] == k]
        ytest = self.y[self.indices == k]
        return np.array(Xtrain),ytrain,np.array(Xtest),ytest

    @property
    def n_class_sample(self):
        n_sample = []
        for y in self.unique_y:
            n_sample.append(sum([x==y for x in self.y]))
        return n_sample

    @property
    def unique_y(self):
        return np.unique(self.y)

def augimage(X,y):
    augX = []
    augy = []
    for i,x in enumerate(X):
        for k in range(4):
            augX.append(np.rot90(x, k))
            augy.append(y[i])

        augX.append(np.fliplr(x))
        augy.append(y[i])
        augX.append(np.flipud(x))
        augy.append(y[i])
        augX.append(np.fliplr(np.rot90(x)))
        augy.append(y[i])
        augX.append(np.flipud(np.rot90(x)))
        augy.append(y[i])
    return np.array(augX),np.array(augy)

def read_images(image_size=256,n_per_class=65,augment=True):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    for i in range(len(names)):
        files = os.walk(path+names[i]+'/')
        files = list(files)[0][2]
        nFile = len(files)
        I = np.random.choice(list(range(nFile)),size=(nFile,),replace=False)
        for j in range(nFile):
            f = files[I[j]]
            im = tf.io.read_file(path + names[i] + '/' + f)
            im = tf.image.decode_jpeg(im, channels=3)
            im = tf.cast(im, tf.float32) / 255.0
            im = tf.image.resize(im, [image_size, image_size])
            if j < n_per_class:
                if augment:
                    im = im.numpy()
                    for k in range(4):
                        train_images.append(np.rot90(im,k))
                        train_labels.append(i)

                    train_images.append(np.fliplr(im))
                    train_labels.append(i)
                    train_images.append(np.flipud(im))
                    train_labels.append(i)
                    train_images.append(np.fliplr(np.rot90(im)))
                    train_labels.append(i)
                    train_images.append(np.flipud(np.rot90(im)))
                    train_labels.append(i)

                else:
                    train_images.append(im.numpy())
                    train_labels.append(i)
            else:
                test_images.append(im.numpy())
                test_labels.append(i)
    n_train = len(train_images)
    I = np.random.choice(list(range(n_train)),n_train,replace=False)
    train_x,train_y,test_x,test_y = np.array(train_images),np.array(train_labels),np.array(test_images),np.array(test_labels)
    return train_x[I],train_y[I],test_x,test_y

def read_images_temporal(image_size=256,n_per_class=65,augment=True):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    for i in range(len(names)):
        files = os.walk(path+names[i]+temporal_tag[0]+'/')
        files = list(files)[0][2]
        nFile = len(files)
        I = np.random.choice(list(range(nFile)),size=(nFile,),replace=False)
        for j in range(nFile):
            f = files[I[j]]
            im = []
            for t in range(len(temporal_tag)):
                im_tmp = tf.io.read_file(path + names[i]+ temporal_tag[t] + '/' + f)
                im_tmp = tf.image.decode_jpeg(im_tmp, channels=3)
                im_tmp = tf.cast(im_tmp, tf.float32) / 255.0
                im_tmp = tf.image.resize(im_tmp, [image_size, image_size])
                im.append(im_tmp)
            im = tf.concat(im,axis=2)
            if j < n_per_class:
                if augment:
                    im = im.numpy()
                    for k in range(4):
                        train_images.append(np.rot90(im,k))
                        train_labels.append(i)
                    train_images.append(np.fliplr(im))
                    train_labels.append(i)
                    train_images.append(np.flipud(im))
                    train_labels.append(i)
                    train_images.append(np.fliplr(np.rot90(im)))
                    train_labels.append(i)
                    train_images.append(np.flipud(np.rot90(im)))
                    train_labels.append(i)

                else:
                    train_images.append(im.numpy())
                    train_labels.append(i)
            else:
                test_images.append(im.numpy())
                test_labels.append(i)
    n_train = len(train_images)
    I = np.random.choice(list(range(n_train)),n_train,replace=False)
    train_x,train_y,test_x,test_y = np.array(train_images),np.array(train_labels),np.array(test_images),np.array(test_labels)
    return train_x[I],train_y[I],test_x,test_y