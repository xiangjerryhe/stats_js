// Principle Component Analysis implementation in javascript using tensorflow.js
import * as tf from "@tensorflow/tfjs";

class PCA {
  constructor(newdim, orig_dim) {
       this.encoder = tf.layers.dense({
                         units: newdim,
                         batchInputShape:[null, orig_dim],
                         kernelInitializer:"randomNormal",
                         biasInitializer:"zeros",
                         activation: 'sigmoid',
                       })
       this.decoder = tf.layers.dense({units:orig_dim, activation:'sigmoid'})
       this.model = tf.sequential({
             layers:[ this.encoder ,  this.decoder] })
        this.model.compile({optimizer:'sgd', loss:'meanSquaredError'})
  }
  
  fit(X) {
         this.model.fit(X, X, {epochs:80, shuffle:true, validationSplit:0.3)  
  }
  transform(X) {
        return this.encoder.predict(X)          
  }
}
