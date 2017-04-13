from cnn.convnet.convnet import ConvNet
from cnn.convnet.recorder import ConvRecorder
from cnn.convnet.utils import init_tf_environ, get_path
from cnn.data.preprocess import IMG_SIZE, CHANNELS, NUM_LABELS, prep_data
from cnn.generate_submission import convnet_submission
from cnn.model import build_model

num_epochs = 10
BATCH_SIZE = 50
EVAL_FREQUENCY = 1


def main():
    init_tf_environ(gpu_num=0)
    all_data = prep_data(test=True)
    model = build_model(*all_data[:4])
    model.restore()
    convnet_submission(model, all_data[4], 'submissions/lenet/submission.csv')
    # model.(BATCH_SIZE, 1, EVAL_FREQUENCY)
    # model.save()

if __name__ == '__main__':
    main()
