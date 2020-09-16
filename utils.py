import numpy as np
import tensorflow as tf

def confusion_matrix(y,yest):
    valid_tag = np.unique(y)
    mat = np.zeros([len(valid_tag),len(valid_tag)])
    for i,true_tag in enumerate(valid_tag):
        for j,pred_tag in enumerate(valid_tag):
            mat[i,j] = np.sum(np.logical_and(y==true_tag, yest==pred_tag))
    mat = mat / np.sum(mat,axis=1,keepdims=True)
    return mat,np.sum(y==yest)/len(y)

def write_config(config,path):
    with open(path,'w+') as f:
        for k in config.keys():
            f.write(f'{k},{config[k].__str__()}\n')

class HSLRSchedular(tf.keras.callbacks.Callback):
    def __init__(self, init_lr, watch_value_name, watch_mode = -1, max_reduce_time=4, reduce_factor=0.25, restart_factor=0.5, patience=10, min_lr = 1e-6, verbose=1):
        super(HSLRSchedular,self).__init__()
        self.init_lr = init_lr
        self.max_reduce_time = max_reduce_time
        self.reduce_factor = reduce_factor
        self.restart_factor = restart_factor
        self.patience = patience

        self.cur_lr = self.init_lr
        self.cur_reduce_time = 0
        self.cur_wait_time = 0
        self.cur_restart_factor = 1
        self.model = None
        self.watch_value_name = watch_value_name
        self.watch_mode = watch_mode
        self.min_lr = min_lr
        if watch_mode > 0:
            self.watch_value = -1e40
        else:
            self.watch_value = 1e40
        self.verbose = verbose

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if self.watch_value_name in logs.keys():
                if (logs[self.watch_value_name] * self.watch_mode) > (self.watch_value * self.watch_mode):
                    if self.verbose:
                        print(f'HSLRSchedular: performance imporved from {self.watch_value} to {logs[self.watch_value_name]}')
                    self.watch_value = logs[self.watch_value_name]
                    self.cur_wait_time = 0
                else:
                    self.cur_wait_time += 1
                    if self.verbose:
                        print(f'HSLRSchedular: performance not imporved, wait time: {self.cur_wait_time}')
                    if self.cur_wait_time >  self.patience:
                        self.cur_reduce_time += 1
                        if self.verbose > 1:
                            print(f'HSLRSchedular: try to reduce, reduce time: {self.cur_reduce_time}')
                        if self.cur_reduce_time > self.max_reduce_time:
                            self.cur_restart_factor *= self.restart_factor
                            if self.verbose > 1:
                                print(f'HSLRSchedular: try to restart, restart factor: {self.cur_restart_factor}')
                            self.cur_lr = self.init_lr * self.cur_restart_factor
                            self.cur_reduce_time = 0
                        else:
                            self.cur_lr *= self.reduce_factor
                        self.cur_wait_time = 0
                self.cur_lr = max(self.cur_lr,self.min_lr)
                if self.verbose:
                    print(f'HSLRSchedular: current learning rate: {self.cur_lr}')
                tf.keras.backend.set_value(self.model.optimizer.lr,self.cur_lr)

            else:
                print(f'HSLRSchedular: no value named {self.watch_value}')
        else:
            print('HSLRSchedular: None logs error')

class HSTensorboard(tf.keras.callbacks.TensorBoard):
    def __init__(self,log_dir,**kwargs):
        super(HSTensorboard, self).__init__(log_dir=log_dir,**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.get_value(self.model.optimizer.lr)})
        super(HSTensorboard, self).on_epoch_end(epoch,logs)