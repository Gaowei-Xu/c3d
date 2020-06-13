# Step 1: download dataset for training and validation
    
    $ cd <workspace>
    $ wget -c https://badminton-video-dataset.s3.amazonaws.com/dataset.zip
    $ unzip dataset.zip


# Step 2: train the model
    
    $ cd src/
    $ python3 train.py