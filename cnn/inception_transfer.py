import tensorflow as tf
import cnn.inception.inception_train as inception_train
# import cnn.inception.inception_eval as inception_eval
from cnn.inception.dataset import CatDogData
from cnn.convnet.utils import get_path, init_tf_environ

FLAGS = tf.app.flags.FLAGS

def main(_):
    init_tf_environ(1)
    FLAGS.data_dir = get_path('data/preprocessed/new')
    FLAGS.train_dir = get_path('models/inception_transfer')
    FLAGS.pretrained_model_checkpoint_path = get_path('models/inception-v3/model.ckpt-157585')
    FLAGS.max_steps = 40000
    FLAGS.fine_tune = True
    FLAGS.subset = 'train'
    FLAGS.initial_learning_rate = 0.005
    FLAGS.num_epochs_per_decay = 5
    FLAGS.learning_rate_decay_factor = 0.9
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
