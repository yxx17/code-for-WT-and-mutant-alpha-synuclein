import numpy as np
import tensorflow as tf
import image_load
import models
import utils
import scipy.io as sio
import time

train_config = {
    'batch_size':64,
    'shuffle_buffer_size':1024,
    'lr':1e-4,
    'early_stop_patience':200,
    'target_path':'data/',
    'model_name':'vgg_mix',
    'Kfold':6,
    'epochs':5000,
    'steps_per_epoch':32,
    'image_resize':128,
    'save_path':'models/mix/',
    'l2_factor':0.025,
    'schedular_watch_name':'val_loss',
    'schedular_max_reduce_time':4,
    'schedular_reduce_factor':0.5,
    'schedular_restart_factor':0.8,
    'schedular_patience':15,
}

def train_model(train_config):
    images,labels = image_load.read_3channel_images(train_config['target_path'],train_config['image_resize'])
    kflod_container = image_load.KFoldContainer(images,labels,train_config['Kfold'])
    utils.write_config(train_config,train_config['save_path']+'config.csv')
    confuse_matrix = []
    accuracy = []
    duration = []

    for k in range(train_config['Kfold']):
        train_x,train_y,test_x,test_y = kflod_container.get_fold_k(k)
        train_x,train_y = image_load.augimage(train_x,train_y)

        print('Load Training {n:d} images'.format(n=len(train_x)))
        print('Load Testing {n:d} images'.format(n=len(test_x)))

        train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
        train_ds = train_ds.shuffle(buffer_size=train_config['shuffle_buffer_size']).repeat().batch(train_config['batch_size'])
        test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
        test_ds = test_ds.repeat(1).batch(train_config['batch_size'])

        model = models.model_7_3(train_config['image_resize'],train_config['l2_factor'])

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=train_config['lr']),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        save_name = f'{train_config["save_path"]}{train_config["model_name"]}-{k}'

        cbs = [tf.keras.callbacks.EarlyStopping(patience=train_config['early_stop_patience']),
               tf.keras.callbacks.ModelCheckpoint(monitor='val_sparse_categorical_accuracy',filepath=save_name,save_best_only=True,save_weights_only=True,verbose=1),
               utils.HSLRSchedular(train_config['lr'],
                                   watch_value_name=train_config['schedular_watch_name'],
                                   max_reduce_time=train_config['schedular_max_reduce_time'],
                                   reduce_factor=train_config['schedular_reduce_factor'],
                                   restart_factor=train_config['schedular_restart_factor'],
                                   patience=train_config['schedular_patience'],
                                   verbose=0),
               utils.HSTensorboard(log_dir=f'./logs/{save_name}/',embeddings_metadata=test_x)]

        ct = time.time()
        model.fit(train_ds,epochs=train_config['epochs'],steps_per_epoch=train_config['steps_per_epoch'],validation_data=test_ds,callbacks=cbs)
        duration.append(time.time()-ct)

        model.load_weights(save_name)
        logits = model.predict(test_x)
        cm,acc = utils.confusion_matrix(test_y,tf.argmax(logits,axis=1).numpy())
        confuse_matrix.append(cm)
        accuracy.append(acc)
        print(f'finish training. k={k}, accuracy={acc:.2f}')

    sio.savemat(train_config['save_path']+'result.mat',{'cm':np.array(confuse_matrix),
                                                        'accuracy':np.array(accuracy),
                                                        'duration':np.array(duration)})



train_sets = ['2h','4h','6h','16h']
for n in train_sets:
    train_config['target_path'] = f'data/pics{n}/'
    train_config['model_name'] = f'vgg{n}'
    train_config['save_path'] = f'models/{n}/'
    train_model(train_config)




