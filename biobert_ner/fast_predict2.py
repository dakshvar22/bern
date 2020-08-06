"""
    Speeds up estimator.predict by preventing it from reloading the graph
    on each call to predict.
    It does this by creating a python generator to keep the predict call open.

    Usage: Just warp your estimator in a FastPredict. i.e.
    classifier = FastPredict(learn.Estimator(model_fn=model_params.model_fn,
    model_dir=model_params.model_dir), my_input_fn)

    This version supports tf 1.4 and above and can be used by
    pre-made Estimators like tf.estimator.DNNClassifier.

    Author: Marc Stogaitis
 """
import tensorflow as tf
import threading
from biobert_ner.utils import Profile


class FastPredict:

    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.first_run = True
        self.closed = False
        self.input_fn = input_fn
        self.next_features = None
        self.predictions = None
        self.lock = threading.Lock()

    def _create_generator(self, batch_size=1):
        while not self.closed:
            # yield self.next_features

            # BioBERT.recognize()
            '''
            for index,f in enumerate(self.next_features):
                # print('yielding', index)
                print(f)
                yield f
            '''
            print('Called with batch size=', batch_size)
            dummy_batch = {'input_ids': [], 'input_mask' : [], 'segment_ids' : [], 'label_ids': []}
            cur_batch = dummy_batch
            cur_batch_size = 0
            for f in self.next_features:
                cur_batch['input_ids'].append(f['input_ids'])
                cur_batch['input_mask'].append(f['input_mask'])
                cur_batch['segment_ids'].append(f['segment_ids'])
                cur_batch['label_ids'].append(f['label_ids'])
                cur_batch_size += 1
                if cur_batch_size == batch_size:
                    print('Yielding new batch of size', cur_batch_size, 'Total next_features', len(self.next_features))
                    yield cur_batch
                    cur_batch = dummy_batch
                    cur_batch_size = 0

            if cur_batch_size:
                print('Yielding new batch of size outside', cur_batch_size, 'Total next_features', len(self.next_features))
                self.next_features=list()
                yield cur_batch
                # raise tf.errors.OutOfRangeError(node_def=None,op=None,message='Empty input')

    @Profile(__name__)
    def predict(self, feature_batch, etype=None):
        """ Runs a prediction on a set of features. Calling multiple times
            does *not* regenerate the graph which makes predict much faster.

            feature_batch a list of list of features.
            IMPORTANT: If you're only classifying 1 thing,
            you still need to make it a batch of 1 by wrapping it in a list
            (i.e. predict([my_feature]), not predict(my_feature)
        """
        with self.lock:
            if self.next_features:
                print('num next features before setting', len(self.next_features), etype)
                
            self.next_features = feature_batch
            print('num next features', len(self.next_features), etype)
            batch_size = len(feature_batch)
            if self.first_run:
                # self.batch_size = len(feature_batch)
                self.predictions = self.estimator.predict(
                    input_fn=self.input_fn(self._create_generator))
                self.first_run = False
            # elif batch_size != len(feature_batch):
            #     raise ValueError(
            #         "All batches must be of the same size. "
            #         "First-batch:" + str(batch_size)
            #         + " This-batch:" + str(len(feature_batch)))

            results = list()
            for _ in range(batch_size):
                results.append(next(self.predictions))
            ''' 
            extra = 0
            while True:
                try:
                    anymore = next(self.predictions)
                    extra += 1
                    if extra % 100 == 0:
                        print('extra', extra)
                    # print('Anymore', anymore)
                except:
                    print('finished', extra)
                    break
            '''
            print('results returned', len(results), etype)
        return results

    def close(self):
        self.closed = True
        try:
            next(self.predictions)
        except Exception as e:
            print(e, "Exception in fast_predict. This is probably OK")


def example_input_fn(generator):
    """ An example input function to pass to predict.
    It must take a generator as input """

    def _inner_input_fn():
        dataset = tf.data.Dataset().from_generator(
            generator, output_types=tf.float32).batch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return {'x': features}

    return _inner_input_fn
