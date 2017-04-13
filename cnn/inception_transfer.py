import tensorflow as tf
import cnn.inception.inception_train as inception_train
# import cnn.inception.inception_eval as inception_eval
from cnn.inception.dataset import CatDogData
from cnn.convnet.utils import get_path

FLAGS = tf.app.flags.FLAGS


def main(_):
    FLAGS.data_dir = get_path('data/preprocessed/new')
    FLAGS.train_dir = get_path('models/inception_transfer')
    FLAGS.pretrained_model_checkpoint_path = get_path('models/inception-v3/model.ckpt-157585')
    FLAGS.max_steps = 100
    FLAGS.fine_tune = True
    FLAGS.subset = 'train'
    dataset = CatDogData(subset=FLAGS.subset)
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    inception_train.train(dataset)


# def evaluate():
#     FLAGS.subset = 'validation'
#     dataset = CatDogData(subset=FLAGS.subset)
#     assert dataset.data_files()
#     if tf.gfile.Exists(FLAGS.eval_dir):
#         tf.gfile.DeleteRecursively(FLAGS.eval_dir)
#     tf.gfile.MakeDirs(FLAGS.eval_dir)
#     inception_eval.evaluate(dataset)

if __name__ == '__main__':
    tf.app.run()