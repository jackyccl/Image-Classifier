# Image-Classifier-Using-Transfer_Learning

In this session, I will run Tensorflow on a single machine and will train a simple classifier to classify images of flowers.
We will be using transfer learning which means we start with a model that has been trained on similar problem. Transfer learning can be done in shorter order as compared to deep learning from scratch.

A model trained on the ImageNet Large Visual Recognition Challenge dataset is being used. 

#### Note: I have uploaded some files to ease your download process, follow each step below on your own to use transfer learning on different categories.


### Step 1: Install Tensorflow and install Docker
The image classifier was tested on Tensorflow version 1.7.0.
```
> pip install --upgrade "tensorflow==1.7.*"
```

### Step 2: Install Docker
Docker is a tool for creating a virtual container on your machine for running apps. The benefit of Docker is that you don't have to install any dependencies on your machine. Eventually, a docker image will has all the necessary dependencies for Tensorflow built in.

### Step 3: Clone the git repository and change directory using ```cd``` into it
```
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
cd tensorflow-for-poets-2
```
Just in case if you download the file on github directly, make sure to just change the directory into the tensorflow-for-poets-2. For example, "C:\Users\User\Desktop\flower_classifier\tensorflow-for-poets-2".

### Step 4: Download the training images
```
curl http://download.tensorflow.org/example_images/flower_photos.tgz \
    | tar xz -C tf_files
```
Inside the dirctory, you can check the copy of flower photos by issuing the following command inside the docker:
```
ls tf_files/flower_photos
```
and the result should shows:
```
daisy/
dandelion/
roses/
sunflowers/
tulip/
LICENSE.txt
```

### Step 5: Retraining the network
As referred to [google tensorflow-for-poets-2 tutorial](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html?index=..%2F..index#3), the MobileNet is configurable in two ways. Hence, we will use 224 pixels and 0.5 relative size of the model as fraction of the largest MobileNet by following command inside the Docker:
```
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"
```

### Step 6: Start TensorBoard
```
tensorboard --logdir tf_files/training_summaries &
```
In Window, make sure that you are in the right directory and just open command prompt under the particular directory. Copy paste the command and you should be able to run tensorboard with the generated link in the cmd.

### Step 7: Run the training on the Docker
```
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos
  ```
  
 The retrain script is from [Tensorflow Hub repo](https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py), you can also run the following script to check each parameter inside the model.
 ```
 python -m scripts.retrain -h
 ```
 
 ### Step 8: Using the Retrained Model
 The retraining script writes data to the following two files:
 *```tf_files/retrianed_graph.pb``` that contains a version of selected network with a final layer retrained on your categories
 *```tf_flies/retrained_labels.txt``` is a text file containing labels
 
 Hence, we can use the retrained model by issueing the following command:
 ```
 python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=(your image directory here)
 ```
 Remember to paste inside your image directory. In Window, you need to change the '\' to '/' if you experience any error for the directory. The result that I get for a daisy photo is shown below:
 
 ```
 daisy (score = 0.99071)
sunflowers (score = 0.00595)
dandelion (score = 0.00252)
roses (score = 0.00049)
tulips (score = 0.00032)
```
This indicates a high confidence (~99%) that the image is a daisy and low confidence for any other label.

#### In the next tutorial, I will be using my own training model to classify cats and dogs.
 
 
 
 
