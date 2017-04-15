from cnn.data.preprocess import prep_data
from cnn.generate_submission import convnet_submission
from cnn.model import build_model
import tensorflow as tf
# import cnn.inception.inception_train as inception_train
import cnn.inception.inception_eval as inception_eval
from cnn.inception.dataset import CatDogData
from cnn.convnet.utils import get_path, init_tf_environ

num_epochs = 10
BATCH_SIZE = 50
EVAL_FREQUENCY = 1

FLAGS = tf.app.flags.FLAGS

def main():
    init_tf_environ(gpu_num=0)
    all_data = prep_data(test=True)
    model = build_model(*all_data[:4])
    model.restore()
    convnet_submission(model, all_data[4], get_path('submissions/inception/submission.csv'))
    # model.(BATCH_SIZE, 1, EVAL_FREQUENCY)
    # model.save()
    FLAGS.subset = 'validation'
    dataset = CatDogData(subset=FLAGS.subset)
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    inception_eval.evaluate(dataset)

if __name__ == '__main__':
    main()
