import tensorflow as tf

from cnn.convnet.convnet import ConvNet
from cnn.convnet.recorder import ConvRecorder
from cnn.convnet.utils import init_tf_environ, get_path
from cnn.data.preprocess import IMG_SIZE, CHANNELS, NUM_LABELS, prep_data, BATCH_SIZE

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('model', 3,
                            """The number of model as defined in the script""")
tf.app.flags.DEFINE_integer('epoch', 30,
                            """The number of epochs to run""")

tf.app.flags.DEFINE_string('name', '',
                           """The name of the model""")

tf.app.flags.DEFINE_string('train', '',
                           """set 'all' if you want to use all the training data""")

# num_epochs = 45
EVAL_FREQUENCY = 1

print('I love you. #heart#')


def build_model(train_data_generator, valid_data_generator):
    models = [model1, model2, model3b, model4, model5, model6]
    model = models[FLAGS.model - 1]()
    model.set_data(train_data_generator, valid_data_generator)
    model.compile()
    return model


def model1():
    # LeNet-5 like Model
    model = ConvNet(FLAGS.name or 'lenet')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[5, 5], out_channels=32, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_conv_layer(filter_size=[5, 5], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=128, activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 1e-4)
    model.set_learning_rate(0.001)
    model.set_optimizer('Adam')
    return model


def model2():
    model = ConvNet(FLAGS.name or 'lenet2')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[7, 7], out_channels=64, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_dropout_layer(0.5)
    model.push_conv_layer(filter_size=[5, 5], out_channels=96, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[5, 5], out_channels=96, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_dropout_layer(0.5)
    model.push_conv_layer(filter_size=[5, 5], out_channels=128, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[5, 5], out_channels=128, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_dropout_layer(0.5)
    model.push_conv_layer(filter_size=[5, 5], out_channels=192, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[5, 5], out_channels=192, strides=[1, 1], activation='relu')
    model.push_pool_layer('avg', kernel_size=[1, int(IMG_SIZE[0]/8), int(IMG_SIZE[1]/8), 1], strides=[int(IMG_SIZE[0]/8), int(IMG_SIZE[1]/8)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=256, activation='relu')
    model.push_dropout_layer(0.5)
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')
    model.set_loss('sparse_softmax')
    # model.set_regularizer('l2', 1e-5)
    model.set_learning_rate(0.001)
    model.set_optimizer('Adam')
    return model


def model3():
    # Network in Network
    model = ConvNet(FLAGS.name or 'NIN')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[7, 7], out_channels=64, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_dropout_layer(0.5)
    model.push_conv_layer(filter_size=[5, 5], out_channels=128, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_dropout_layer(0.5)
    model.push_conv_layer(filter_size=[5, 5], out_channels=256, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=256, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=256, strides=[1, 1], activation='relu')
    model.push_pool_layer('avg', kernel_size=[1, int(IMG_SIZE[0]/4), int(IMG_SIZE[1]/4), 1], strides=[int(IMG_SIZE[0]/4), int(IMG_SIZE[1]/4)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=512, activation='relu')
    model.push_dropout_layer(0.7)
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.set_loss('sparse_softmax')
    # model.set_regularizer('l2', 1e-5)
    model.set_learning_rate(0.002)
    model.set_optimizer('Adam')
    return model


def model3b():
    # Network in Network
    model = ConvNet(FLAGS.name or 'NIN-test')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[5, 5], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=64, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_conv_layer([5, 5], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=128, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_conv_layer([5, 5], out_channels=256, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=256, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=256, strides=[1, 1], activation='linear')
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[1, int(IMG_SIZE[0]/4), int(IMG_SIZE[1]/4), 1], strides=[int(IMG_SIZE[0]/4), int(IMG_SIZE[1]/4)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=512, activation='relu')
    model.push_dropout_layer(0.7)
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 1e-5)
    model.set_learning_rate(0.001)
    model.set_optimizer('Adam')
    return model


def model4():
    # test resnet
    model = ConvNet(FLAGS.name or 'ResNet')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[3, 3], out_channels=16, strides=[2, 2], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_res_layer([3, 3], 64, strides=[1, 1], activate_before_residual=False, activation='relu')
    for i in range(5):
        model.push_res_layer([3, 3], 64, strides=[1, 1], activation='relu')
    model.push_res_layer([3, 3], 128, strides=[2, 2], activation='relu')
    for i in range(5):
        model.push_res_layer([3, 3], 128, strides=[1, 1], activation='relu')
    model.push_res_layer([3, 3], 256, strides=[2, 2], activation='relu')
    for i in range(4):
        model.push_res_layer([3, 3], 256, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)],
                          strides=[int(IMG_SIZE[0] / 8), int(IMG_SIZE[1] / 8)])
    model.push_flatten_layer()
    # model.push_fully_connected_layer(512, activation='linear', has_bias=True)
    model.push_fully_connected_layer(NUM_LABELS, activation='linear', has_bias=True)
    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 1e-5)
    # model.set_learning_rate(0.02, 'exponential', decay_rate=0.95)
    # model.set_optimizer('Momentum', momentum=0.9)
    model.set_learning_rate(0.0002) # 0.0005
    model.set_optimizer('Adam', beta2=0.99)
    return model


def model5():
    # test resnet
    model = ConvNet(FLAGS.name or 'ResNet2')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[7, 7], out_channels=64, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_res_layer([3, 3], 64, strides=[1, 1], activation='relu')
    for i in range(2):
        model.push_res_layer([3, 3], 64, strides=[1, 1], activation='relu')
    model.push_res_layer([3, 3], 128, strides=[2, 2], activation='relu')
    for i in range(3):
        model.push_res_layer([3, 3], 128, strides=[1, 1], activation='relu')
    model.push_res_layer([3, 3], 256, strides=[2, 2], activation='relu')
    for i in range(5):
        model.push_res_layer([3, 3], 256, strides=[1, 1], activation='relu')
    model.push_res_layer([3, 3], 512, strides=[2, 2], activation='relu')
    for i in range(2):
        model.push_res_layer([3, 3], 512, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 16), int(IMG_SIZE[1] / 16)],
                          strides=[int(IMG_SIZE[0] / 16), int(IMG_SIZE[1] / 16)])
    model.push_flatten_layer()
    # model.push_fully_connected_layer(512, activation='linear', has_bias=True)
    model.push_fully_connected_layer(NUM_LABELS, activation='linear', has_bias=True)
    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 1e-5)
    model.set_learning_rate(0.0002)
    model.set_optimizer('Adam')
    return model


def model6():
    # test resnet
    model = ConvNet(FLAGS.name or 'ResNet2')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[7, 7], out_channels=64, strides=[1, 1], activation='linear', has_bias=False)
    model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('max', [2, 2], strides=[2, 2])
    model.push_res_layer([3, 3], 64, strides=[1, 1], activation='relu')
    for i in range(2):
        model.push_res_layer([3, 3], 64, strides=[1, 1], activation='relu')
    model.push_res_layer([3, 3], 128, strides=[2, 2], activation='relu')
    for i in range(3):
        model.push_res_layer([3, 3], 128, strides=[1, 1], activation='relu')
    model.push_res_layer([3, 3], 256, strides=[2, 2], activation='relu')
    for i in range(5):
        model.push_res_layer([3, 3], 256, strides=[1, 1], activation='relu')
    model.push_res_layer([3, 3], 512, strides=[2, 2], activation='relu')
    for i in range(2):
        model.push_res_layer([3, 3], 512, strides=[1, 1], activation='relu')
    # model.push_batch_norm_layer(activation='relu')
    model.push_pool_layer('avg', kernel_size=[int(IMG_SIZE[0] / 16), int(IMG_SIZE[1] / 16)],
                          strides=[int(IMG_SIZE[0] / 16), int(IMG_SIZE[1] / 16)])
    model.push_flatten_layer()
    # model.push_fully_connected_layer(512, activation='linear', has_bias=True)
    model.push_fully_connected_layer(NUM_LABELS, activation='linear', has_bias=True)
    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 1e-5)
    model.set_learning_rate(0.01, 'piecewise_constant', boundaries=[20000, 32000, 44000, 55000],
                            values=[0.01, 0.002, 0.0004, 0.0001, 0.00002])
    model.set_optimizer('Momentum', momentum=0.9)
    # model.set_learning_rate(0.0002)
    # model.set_optimizer('Adam')
    return model


def main():
    init_tf_environ(gpu_num=1)
    all_data = prep_data(test=False, all=FLAGS.train == 'all')
    model = build_model(*all_data[:2])
    # rec = ConvRecorder(model, get_path('models', 'lenet/train'))
    model.train(BATCH_SIZE, FLAGS.epoch, EVAL_FREQUENCY)
    model.save()


if __name__ == '__main__':
    main()
