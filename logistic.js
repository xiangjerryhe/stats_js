import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

class Logistic {
    constructor(featureCount) {
      this.model = tf.sequential({
        layers:[tf.layers.dense({
          units:2,
          activation:"softmax",
          inputShape:[featureCount]
        })]
      })
      this.model.compile({
        optimizer:tf.train.adam(0.001),
        loss:"binaryCrossentropy",
        metrics:["accuracy"]
      })
    }
    get model() {
        return this.model
    }
     fit(X) {
       const trainLogs = [];
       await this.model.fitDataset(X, {
         epochs:100, 
         callbacks:{
           onEpochEnd: async (epoch, logs) => {
             trainLogs.push(logs)
             tfvis.show.history(this_div(), trainLogs, ["acc"])
           }
         }
       })
     }
}
