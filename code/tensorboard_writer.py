import os
import logging
import tensorflow as tf

from datetime import datetime
from tensorboard import default, program


# writes data to tensorboard file
class TensorboardWriter:
    DIR = './tensorboard'

    # start tensorboard server
    def run():
        # Remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)

        # Start tensorboard server
        tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
        tb.configure(argv=[None, '--logdir', TensorboardWriter.DIR])
        url = tb.launch()
        print('TensorBoard at %s \n' % url)

    # setup file and placeholders for tensorboard
    def __init__(self, store_dir, stats = []):
        store_dir = os.path.join(TensorboardWriter.DIR, store_dir)
        os.makedirs(store_dir, exist_ok=True)

        tf.reset_default_graph()
        self.sess = tf.Session()

        self.tf_writer = tf.summary.FileWriter(os.path.join(store_dir, "experiment-%s" % datetime.now().strftime("%Y%m%d-%H%M%S") ))

        self.stats = stats
        self.pl_stats = {}

        for s in self.stats:
            self.pl_stats[s] = tf.placeholder(tf.float32, name=s)
            tf.summary.scalar(s, self.pl_stats[s])

        self.performance_summaries = tf.summary.merge_all()

    # write episode statistics in eval_dict to tensorboard
    def write_episode_data(self, episode, eval_dict):
       my_dict = {}
       for k in eval_dict:
          assert(k in self.stats)
          my_dict[self.pl_stats[k]] = eval_dict[k]

       summary = self.sess.run(self.performance_summaries, feed_dict=my_dict)

       self.tf_writer.add_summary(summary, episode)
       self.tf_writer.flush()

    # close session
    def close_session(self):
        self.tf_writer.close()
        self.sess.close()