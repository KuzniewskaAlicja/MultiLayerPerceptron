# MultiLayerPerceptron

Project contains implementation of simple multi-layer perceptron for classification task without using any additional libraries except numpy.
Implementation includes:
  - Generation simple synthetized dataset
  - Splitting data into train and validation datasets, normalization and generating batches
  - Model implementation with feedforward and backpropagation
  - Training loop
  - Saving weights for which speicified metric was the best based on monitoring mode (minimalization, maximalization)
  - Loading saved weights

1. Install dependencies
   ```
     pip install -r requirements.txt
   ```
3. Running training script
   ```
     python main.py [parameters]
   ```
   Parameters:
   |Name|Default|Description|
   |:--|:-----|:---------|
   |early_stopping|None|Number of epochs to wait until stop training if there is no progress|
   |name|model|Model name used as file name of model weights|
   |model_weights|None|Path to weights file, if specified weights will be loaded to a model|

5. Results overview

   Created dataset details:
   - samples: 500
   - features: 10
   - classes: 3

   Model details:
   ```
     MultiLayerPerceptron(
	      [0] InputLayer(10)
	      [1] HiddenLayer(10 -> 16) activation=ReLU
	      [2] HiddenLayer(16 -> 8) activation=ReLU
	      [3] OutputLayer(8 -> 3) activation=Softmax
     )

   ```

   Training details:
   - training epoch number: 20
   - learning rate: 0.001
   - loss function: CategoricalCrossEntropy
   - metric: Accuracy
   - batch_size:
     - val: 64
     - train: 32
   - dataset split with val factor set to 0.2 with data shuffle

   Obstained metrics:
   |Dataset|Metric name|Matric value|
   |:------:|:---------:|:-----------|
   |Train|loss|0.01433|
   ||accuracy|1.0|
   |Validation|loss|0.01463|
   ||accuracy|1.0|
   
