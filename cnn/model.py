from cnn.convnet.convnet import ConvNet
from cnn.convnet.recorder import ConvRecorder
from cnn.preprocess import IMG_SIZE, CHANNELS, NUM_LABELS, prep_data

num_epochs = 10
BATCH_SIZE = 50
EVAL_FREQUENCY = 1


def build_model():

    all_data = prep_data(test=False)
    train_data = all_data[0][:1000]
    train_labels = all_data[1][:1000]
    valid_data = all_data[2]
    valid_labels = all_data[3]
    test_data = all_data[4]
    # LeNet-5 like Model
    model = ConvNet('mnist')
    model.push_input_layer(dshape=[None, IMG_SIZE[0], IMG_SIZE[1], CHANNELS])
    model.push_conv_layer(filter_size=[5, 5], out_channels=32, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_conv_layer(filter_size=[5, 5], out_channels=64, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=512, activation='relu')
    model.push_dropout_layer(0.5)
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 5e-4)
    model.set_learning_rate(0.001)
    model.set_optimizer('Adam')
    model.set_data(train_data, train_labels, valid_data, valid_labels)
    model.compile()
    rec = ConvRecorder(model, '../model/minist/train')
    model.train(BATCH_SIZE, 50, EVAL_FREQUENCY)


if __name__ == '__main__':
    build_model()
