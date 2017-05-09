import tensorflow as tf

from cnn.convnet.convnet import ConvNet
from cnn.convnet.recorder import ConvRecorder
from cnn.convnet.utils import init_tf_environ, get_path, before_save
from cnn.preprocess import IMG_SIZE, CHANNELS, NUM_LABELS, prepare_data_fer2013, BATCH_SIZE, TRAIN_SIZE
from cnn.generate_submission import lists2csv

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('model', 1,
                            """The number of model as defined in the script""")
tf.app.flags.DEFINE_integer('epoch', 30,
                            """The number of epochs to run""")

tf.app.flags.DEFINE_string('name', '',
                           """The name of the model""")

tf.app.flags.DEFINE_string('train', '',
                           """set 'all' if you want to use all the training data""")

# num_epochs = 45
EVAL_FREQUENCY = 1
N = TRAIN_SIZE

def build_model(model_no, name, train_data_generator, valid_data_generator):
    models = [model1, model2, model3, model4, model5]
    model = models[model_no - 1](name)
    model.set_data(train_data_generator, valid_data_generator)
    model.compile()
    return model


def model1(name=''):
    # Network in Network
    model = ConvNet(name or 'NIN-test')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_augment_layer(4, 4, True, True)
    model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_conv_layer(filter_size=[3, 3], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', kernel_size=[2, 2], strides=[2, 2])
    model.push_conv_layer([3, 3], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0]/4), int(IMG_SIZE[1]/4)], strides=[int(IMG_SIZE[0]/4), int(IMG_SIZE[1]/4)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 5e-3)
    model.set_learning_rate(0.001, 'piecewise_constant', boundaries=[10*N//BATCH_SIZE, 15*N//BATCH_SIZE, 20*N//BATCH_SIZE], 
                            values=[0.1, 0.01, 0.001, 0.0001])
    model.set_optimizer('Momentum', 0.9)
    # model.set_learning_rate(0.001)
    # model.set_optimizer('Adam')
    return model


def model2(name=''):
    # Network in Network
    model = ConvNet(name or 'NIN2')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[2, 2], activation='linear', has_bias=False) 
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_conv_layer(filter_size=[3, 3], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', kernel_size=[2, 2], strides=[2, 2])
    model.push_conv_layer([3, 3], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0]/8), int(IMG_SIZE[1]/8)], strides=[int(IMG_SIZE[0]/8), int(IMG_SIZE[1]/8)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 5e-3)
    model.set_learning_rate(0.001, 'piecewise_constant', boundaries=[20*N//BATCH_SIZE, 30*N//BATCH_SIZE, 40*N//BATCH_SIZE], 
                            values=[0.1, 0.01, 0.001, 0.0001])
    model.set_optimizer('Momentum', 0.9)
    return model


def model3(name=''):
    # test resnet
    model = ConvNet(name or 'ResNet')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    # model.push_conv_layer(filter_size=[7, 7], out_channels=64, strides=[1, 1], activation='linear', has_bias=False)
    model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    # model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_res_layer([3, 3], 32, strides=[1, 1], activation='relu', activate_before_residual=False)
    for i in range(2):
        model.push_res_layer([3, 3], 32, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_res_layer([3, 3], 64, strides=[2, 2], activation='relu', activate_before_residual=False)
    for i in range(3):
        model.push_res_layer([3, 3], 64, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_res_layer([3, 3], 128, strides=[2, 2], activation='relu', activate_before_residual=False)
    for i in range(5):
        model.push_res_layer([3, 3], 128, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_res_layer([3, 3], 256, strides=[2, 2], activation='relu', activate_before_residual=False)
    for i in range(2):
        model.push_res_layer([3, 3], 256, strides=[1, 1], activation='relu')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)],
                          strides=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(NUM_LABELS, activation='linear', has_bias=True)
    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 2e-3)
    model.set_learning_rate(0.01, 'piecewise_constant',
                            boundaries=[10 * N // BATCH_SIZE, 15 * N // BATCH_SIZE, 20 * N // BATCH_SIZE],
                            values=[0.1, 0.01, 0.001, 0.0001])
    model.set_optimizer('Momentum', momentum=0.9)
    # model.set_learning_rate(0.1)  # 0.001 for RMSProp
    # model.set_optimizer('Adadelta')
    return model


def model4(name=''):
    # test resnet
    model = ConvNet(name or 'ResBN')
    # model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    # model.push_conv_layer(filter_size=[7, 7], out_channels=32, strides=[2, 2], activation='linear', has_bias=False)
    # model.push_batch_norm_layer(activation='relu')
    # model.push_pool_layer('max', [2, 2], strides=[2, 2])
    # # model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[1, 1], activation='linear', has_bias=False)
    # # model.push_batch_norm_layer(activation='relu')
    # # model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    # # model.push_batch_norm_layer(activation='relu')
    # # model.push_pool_layer('max', [2, 2], strides=[2, 2])
    # model.push_res_bn_layer([3, 3], 64, strides=[1, 1], activation='relu', activate_before_residual=False)
    # for i in range(2):
    #     model.push_res_bn_layer([3, 3], 64, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    # model.push_res_bn_layer([3, 3], 128, strides=[2, 2], activation='relu', activate_before_residual=False)
    # for i in range(3):
    #     model.push_res_bn_layer([3, 3], 128, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    # model.push_res_bn_layer([3, 3], 256, strides=[2, 2], activation='relu', activate_before_residual=False)
    # for i in range(5):
    #     model.push_res_bn_layer([3, 3], 256, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    # model.push_res_bn_layer([3, 3], 512, strides=[2, 2], activation='relu', activate_before_residual=False)
    # for i in range(2):
    #     model.push_res_bn_layer([3, 3], 512, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    # model.push_pool_layer('avg', kernel_size=[8, 8],
    #                       strides=[8, 8])
    # model.push_flatten_layer()
    # # model.push_fully_connected_layer(512, activation='linear', has_bias=True)
    # model.push_fully_connected_layer(NUM_LABELS, activation='linear', has_bias=True)
    # model.set_loss('sparse_softmax')
    # model.set_regularizer('l2', 1e-5)
    # model.set_learning_rate(0.01, 'piecewise_constant', boundaries=[40*N//BATCH_SIZE, 60*N//BATCH_SIZE, 75*N//BATCH_SIZE],
    #                         values=[0.1, 0.01, 0.001, 0.0001])
    # #                         values=[0.02, 0.002, 0.0002, 0.00002])  # boundaries=[40000, 52000, 64000]
    # model.set_optimizer('Momentum', momentum=0.9)
    # model.set_learning_rate(0.0002)
    # model.set_optimizer('Adam')
    return model


def model5(name=''):
    # test resnet
    model = ConvNet(name or 'ResNet2')
    # model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    # model.push_conv_layer(filter_size=[7, 7], out_channels=64, strides=[2, 2], activation='linear', has_bias=False)
    # model.push_batch_norm_layer(activation='relu')
    # model.push_pool_layer('max', [2, 2], strides=[2, 2])
    # # model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[1, 1], activation='linear', has_bias=False)
    # # model.push_batch_norm_layer(activation='relu')
    # # model.push_conv_layer(filter_size=[3, 3], out_channels=32, strides=[1, 1], activation='linear', has_bias=False)
    # # model.push_batch_norm_layer(activation='relu')
    # # model.push_pool_layer('max', [2, 2], strides=[2, 2])
    # model.push_res_layer([3, 3], 64, strides=[1, 1], activation='relu', activate_before_residual=False)
    # for i in range(2):
    #     model.push_res_layer([3, 3], 64, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    # model.push_res_layer([3, 3], 128, strides=[2, 2], activation='relu', activate_before_residual=False)
    # for i in range(3):
    #     model.push_res_layer([3, 3], 128, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    # model.push_res_layer([3, 3], 256, strides=[2, 2], activation='relu', activate_before_residual=False)
    # for i in range(5):
    #     model.push_res_layer([3, 3], 256, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    # model.push_res_layer([3, 3], 512, strides=[2, 2], activation='relu', activate_before_residual=False)
    # for i in range(2):
    #     model.push_res_layer([3, 3], 512, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    # model.push_pool_layer('avg', kernel_size=[8, 8],
    #                       strides=[8, 8])
    # model.push_flatten_layer()
    # # model.push_fully_connected_layer(512, activation='linear', has_bias=True)
    # model.push_fully_connected_layer(NUM_LABELS, activation='linear', has_bias=True)
    # model.set_loss('sparse_softmax')
    # model.set_regularizer('l2', 1e-5)
    # model.set_learning_rate(0.01, 'piecewise_constant', boundaries=[40*N//BATCH_SIZE, 60*N//BATCH_SIZE, 75*N//BATCH_SIZE],
    #                         values=[0.1, 0.01, 0.001, 0.0001])
    # #                         values=[0.02, 0.002, 0.0002, 0.00002])  # boundaries=[40000, 52000, 64000]
    # model.set_optimizer('Momentum', momentum=0.9)
    # model.set_learning_rate(0.0002)
    # model.set_optimizer('Adam')
    return model


def eval(model, data_generator):

    loss, acc, acc3 = model.eval(model.sess, data_generator, BATCH_SIZE)
    print('[Test Set] Loss: {:.3f}, Acc: {:.2f%}, Acc3: {:.2f}, eval num: {:d}'.format(
          loss, acc, acc3, data_generator.n // data_generator.batch_size * data_generator.batch_size))

def main():
    init_tf_environ(gpu_num=1)
    all_data = prepare_data_fer2013()
    model = build_model(FLAGS.model, FLAGS.name, all_data['train'], all_data['valid'])
    # rec = ConvRecorder(model, get_path('models', 'lenet/train'))
    losses, valid_losses = model.train(BATCH_SIZE, FLAGS.epoch, EVAL_FREQUENCY)
    model.save()
    total_steps = FLAGS.epoch * all_data['train'].n // BATCH_SIZE
    log_step = all_data['train'].n // BATCH_SIZE // 10
    train_steps = range(0, total_steps, log_step)
    valid_steps = range(0, total_steps, log_step * 10)
    train_log_file = get_path('log', model.name_or_scope + '_train.csv')
    valid_log_file = get_path('log', model.name_or_scope + '_valid.csv')
    before_save(train_log_file)
    lists2csv(list(zip(train_steps, losses)), train_log_file, header=['step', 'loss'])
    lists2csv(list(zip(valid_steps, valid_losses)), valid_log_file, header=['step', 'loss'])
    eval(model, all_data['test'])


if __name__ == '__main__':
    main()
