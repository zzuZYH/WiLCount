# -*- coding=utf-8 -*-
import time
import keras
from load_data import load_images_data
from WiLCount import WiLCount as build_model


from read_data_modify import *
from utils.utils import *
from arguments import *
from sklearn.model_selection import train_test_split
from collections import Counter



# 添加GPU
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_n

for data_name in ALL_DATA_NAMES:
    print('正在训练-- ', data_name, ' --数据集')
    # --------------------读取数据-------------------------
    xx1, yy1 = load_images_data(INPUT_PROCESSED_DATA_PKG+data_name+'/am',type="all")
    xx2, yy2 = load_images_data(INPUT_PROCESSED_DATA_PKG + data_name+'/phase', type="all")
    print(xx1.shape)
    print(xx2.shape)
    xx = np.concatenate((xx1,xx2),axis=-1)
    print(xx.shape)
    yy = yy1

    activity_list = list(Counter(yy).keys())
    print('(数据标签)：', activity_list)
    yy = onehot(yy)
    print(yy)

    # --------------------数据处理-------------------------
    shape = xx.shape
    print("--xx.shape:",xx.shape)
    print("--yy.shape:", yy.shape)


    # --------------------数据分割-------------------------
    x_train,x_test, y_train, y_test = train_test_split(xx, yy, test_size = kk/100, random_state = 1)
    print('--activity_list: ', activity_list)

    # --------------------以下为训练部分-------------------------
    y_true_val = None
    y_pred_val = None
    y_true = np.argmax(y_test, 1)

    start_time = time.time()
    input_shape = x_train.shape[1:]

    print('input_shape', input_shape)
    nb_classes = len(activity_list)
    print('nb_classes: ', nb_classes)
    #
    model = build_model(input_shape=input_shape, nb_classes=nb_classes)

    # reduce learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                min_lr=0.001)
    # model checkpoint
    now = int(time.time())
    timeArray = time.localtime(now)
    timestr = time.strftime("_%m%d_%H%M%S_", timeArray)
    save_name = timestr + data_name + testname

    ckp_metric_file_path = result_save_path + save_name + '/'
    ckp_metric_file_path = create_directory(ckp_metric_file_path)
    print('模型输出文件夹：', ckp_metric_file_path)

    save_model_path = ckp_metric_file_path + 'model-io-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}-val_acc{val_accuracy:.5f}.h5'

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_accuracy',
                                                    save_best_only=True, verbose=1, mode='max', period=5)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=ckp_metric_file_path + '/log/')
    callbacks = [reduce_lr, model_checkpoint, tensorboard]

    if verbose is True:
        model.summary()

    # --------------------训练模型-------------------------
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, shuffle=True,
                     verbose=verbose, validation_split=0.2, callbacks=callbacks)
    save_hist(ckp_metric_file_path, hist, lr=True)


    # --------------------评估模型-------------------------
    best_val_model_path = choose_best_model(result_save_path + save_name)
    model = keras.models.load_model(best_val_model_path)

    y_pred_b = model.predict(x_test)
    y_pred = np.argmax(y_pred_b, axis=1)

    duration = time.time() - start_time

    save_best_path(ckp_metric_file_path)
    df_metrics = save_metrics(ckp_metric_file_path, y_pred, y_true, duration, y_true_val, y_pred_val)
    save_confusion(y_true, y_pred, activity_list, save_path=ckp_metric_file_path+'/')
    save_result_binary(y_test, y_pred_b, activity_list, save_path=ckp_metric_file_path+'/')


    print(df_metrics)

tf.keras.backend.clear_session()
