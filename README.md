## C3D Network for Badminton Action Sequence Classification

#### Step 1: download dataset for training and validation
    
    $ cd <workspace>
    $ wget -c https://badminton-video-dataset.s3.amazonaws.com/dataset.zip
    $ unzip dataset.zip
    
samples amount distribution over 18 classes

![avatar] (https://github.com/Gaowei-Xu/c3d/blob/master/dist.jpg)



#### Step 2: Install dependencies
    
    $ cd <workspace>
    $ sudo pip3 install -r requirements.txt

#### Step 3: Train the model
    
    $ cd src/
    $ python3 train.py


#### Train log could be available in train.log

Initial pre-trained model receives ~38% accuracy over validation dataset.
