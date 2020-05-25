import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('./model/model.json');

function predict(img) {
  return model.predict(img);
}
