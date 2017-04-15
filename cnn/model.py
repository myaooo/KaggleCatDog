from cnn.convnet.convnet import ConvNet
from cnn.convnet.recorder import ConvRecorder
from cnn.convnet.utils import init_tf_environ, get_path
from cnn.data.preprocess import IMG_SIZE, CHANNELS, NUM_LABELS, prep_data

num_epochs = 10
BATCH_SIZE = 20
EVAL_FREQUENCY = 1

print('I love you. #heart#')


def build_model(train_data, train_labels, valid_data, valid_labels):
    # LeNet-5 like Model
    model = model2()
    model.set_data(train_data, train_labels, valid_data, valid_labels)
    model.compile()
    return model


def model1():
    model = ConvNet('lenet')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[5, 5], out_channels=64, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_conv_layer(filter_size=[5, 5], out_channels=64, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=128, activation='relu')
    model.push_dropout_layer(0.5)
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 1e-3)
    model.set_learning_rate(0.001)
    model.set_optimizer('Adam')
    return model


def model2():
    model = ConvNet('lenet2')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[5, 5], out_channels=128, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_conv_layer(filter_size=[6, 6], out_channels=128, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_dropout_layer(0.5)
    model.push_conv_layer(filter_size=[7, 7], out_channels=128, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=128, activation='relu')
    model.push_dropout_layer(0.5)
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 5e-4)
    model.set_learning_rate(0.002)
    model.set_optimizer('Adam')
    return model

def model3():
# Network in Network
    model = ConvNet('NIN')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[6, 6], out_channels=192, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=192, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=192, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_dropout_layer(0.5)
    model.push_conv_layer(filter_size=[6, 6], out_channels=192, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=192, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=192, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_dropout_layer(0.5)
    model.push_conv_layer(filter_size=[4, 4], out_channels=192, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=192, strides=[1, 1], activation='relu')
    model.push_conv_layer(filter_size=[1, 1], out_channels=NUM_LABELS, strides=[1, 1], activation='relu')
    model.push_pool_layer('avg', kernel_size=[1, int(IMG_SIZE[0]/4), int(IMG_SIZE[1]/4), 1], strides=[int(IMG_SIZE[0]/4), int(IMG_SIZE[1]/4)])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=128,activation='relu')
    model.push_dropout_layer(0.5)
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.set_loss('sparse_softmax')
    # model.set_regularizer('l2', 6e-5)
    model.set_learning_rate(0.002)
    model.set_optimizer('Adam')
    return model

def main():
    init_tf_environ(gpu_num=1)
    all_data = prep_data(test=False)
    model = build_model(*all_data[:4])
    rec = ConvRecorder(model, get_path('models', 'lenet/train'))
    model.train(BATCH_SIZE, 20, EVAL_FREQUENCY)
    model.save()


if __name__ == '__main__':
    main()
