# This is a easy to use captcha tool in pytorch.
By Wang Zihmin 2019

## description:

There was a captcha generation tool at https://github.com/JackonYang/captcha-tensorflow. However, the model was only in tensorflow and it hasn't been updated for a while, some of the code in it has been depreciated.

Thus, I wrote a pytorch captcha solver. It has an easy to use interface for naive user and some functionalities to play with for more advanced use. I borrowed the captcha generation tool from the above link and slightly modify it to accommodate more needs. Depending on the differences between training data and images to be predicted, the performance can be quite bad actually. Fetching data that is generated from the source that you want to predict if high accuracy rate is what you need. Training a general model can be challenging due to the simplicity of this model. Maybe adding some attention and dropout would help.

## try in command line to generate a small example:

    python3 ./gen_captcha.py -s 100 --npi 2 -d
    python3 ./cnn_captcha.py -d ./images/char-2-sample-100 -e 20
    python3 ./cnn_captcha.py -p -i ./images/char-2-sample-100/test/<please select a image> -l ./images/char-2-sample-100/20_checkpoint.tar

try to generate a larger dataset or fetch one by your self to get better performance. See indivisual usage below.

## gen_captcha.py usage:

This is a program used to generate captcha dataset. Please refer to the above link for more detailed usage.
  
Parameters -s -height -width are added to the original file. 
  

    -h, --help           show this help message and exit
    -s S                 the number of captchas to generate for training.
    --height HEIGHT      number of pixel in height.
    --width WIDTH        number of pixel in width.
    -t T                 ratio of test dataset. default to 0.2. -s * -t captchas
                       will be generated for testing
    -d, --digit          use digits in dataset.
    -l, --lower          use lowercase in dataset.
    -u, --upper          use uppercase in dataset.
    --npi NPI            number of characters per image.
    --data_dir DATA_DIR  where data will be saved.


  * command line
  

        
        e.g. python3 ./gen_captcha.py -s 100 --npi 5 -d --width 140 --height 45 
    This would generate 100 images of size height 45 * width 140 containing 5 digits in the directory /char-5-sample-10/train in a default path ./image. 
    
    It would also generate a testing dataset and a meta.json. All image names are prefixed with answers. Try running the above little example to see the details. 
        

## cnn_captcha.py usage: 
This is a program used to train model and predict images.

    -h, --help            show this help message and exit
    -d DATA, --data DATA  A path to the directory containing training data
                        (images).
    -b BATCHSIZE, --batchSize BATCHSIZE
                        batch size
    -l LOADCHECK, --loadCheck LOADCHECK
                        path to a checkpoint. -h -w --npi of gen_captcha and
                        --convLayer --convKernel --fcLayer if specified
                        explicitly, if you are using default then it will be
                        ok, in this file should be the same as checkpoint, for
                        we are loading both conv and fc
    -e EPOCH, --epoch EPOCH
                        total number of epoch for either new model or resumed
                        model. e.g. -e 30 -l 15_checkpoint.tar would train
                        this model for 15 more epoches.
    -p, --predict         Predict mode.The image to be predicted should have the
                        same width and height of the training data. prediction
                        mode ignore all parameters except -i and -l and
                        itself.
    -i IMAGE, --image IMAGE
                        A path to "an" image to precict. It accept only an
                        image at a time now
    --printEveryBatch PRINTEVERYBATCH
                        print loss every this number of unit.
    --learnRate LEARNRATE
                        learning rate for the optimizer
    --weightDecay WEIGHTDECAY
                        L2 regulizer
    --grayScale           Load in all training images as gray scale images. When
                        using this mode all the following operation should and
                        will be performed in this mode.
    --convLayer CONVLAYER
                        number of layers for convolution network. More layers
                        may be able to capture a larger pattern, especially
                        when the image size is big.
    --convKernel CONVKERNEL
                        size of the kernel for all convolution network. Larger
                        kernel may be able to capture a larger pattern in a
                        image, especially when the image size is big.
    --fcLayer FCLAYER     number of layers for fully connected network. It
                        results in a bigger model.
    --pretrainedModel PRETRAINEDMODEL
                        load pretrained convolution Model, for captcha with
                        many letters are hard to train directly. e.g. load a 2
                        digit checkpoint to train a 5 digit model. -w -h of
                        gen_captcha.py and --convLayer --convKernel if
                        specified explicitly, if you are using default then it
                        will be ok, should be the same as the old one, for we
                        are loading a conv model.
    --fixConv             fix the parameters in convLayers.


### train model:
    

* command line
 
    *  easy usage:
            
            possible parameter -d -l -e -b
            required parameter -d

            
            e.g. python3 ./cnn_captcha.py -d ./images/char-2-sample-10
            e.g. python3 ./cnn_captcha.py -d ./images/char-2-sample-10 -l ./images/char-2-sample-1/30_checkpoint.tar -e 40 -b 16
    These command would automatically generate a checkpoint file in the directory passed to -d that can be used to predict images, see below for the prediction part of this program, resume training, and pretrain a model, see examples or the above description on how to use these functionalities.
     * more advanced usage:

            possible parameter -d -l -e -b --printEveryBatch --learnRate --weightDecay --grayScale --convLayer --convKernel --fcLayer --pretrainedModel --fixConv
            required parameter -d

            e.g. python3 cnn_captcha.py -d images/char-5-sample-10/ --convLayer 3 --convKernel 7 --fcLayer 4
            e.g. python3 cnn_captcha.py -d images/char-5-sample-10/ --learnRate 0.0001 -b 16 --printEveryBatch 1 
            e.g. python3 cnn_captcha.py -d images/char-5-sample-10/ --pretrainedModel ./images/char-2-sample-10/30_checkpoint.tar --fixConv -e 45 --fcLayer 4
            e.g. python3 cnn_captcha.py -d images/char-1-sample-5/ --grayScale --convLayer 3 --fcLayer 3 -e 10 --printEveryBatch 100
	
* note:
         
    A larger dataset and appropriate parameter settings, e.g. batchSize, can result in better performance. This model can also take in training data generated from other sourses. But in order to train this model from them, Please follow the dataset structure generated by gen_captcha.py.
    
    Parameter -d of cnn_captcha.py is passed a path to a directory that contains a meta file named meta.json, a directory named /test containing testing images, and a directory named /train containing training images. Try the above small example to see their contents in detail. All images should have the same size and have names prefixed by answers for that image.
  


### predict captcha:
            

* command line:

        possible parameter -p -i -l
        required parameter -p -i -l
        
        e.g. python3 ./cnn_captcha.py -p -i ./images/char-2-sample-10/test/<fileNameOfAnImage> -l ./images/char-2-sample-10/30_checkpoint.tar
    Pass -i a path to an image file. After running this line you should see the predicted answer at stdout.
            
* function call:
        
        import cnn_captcha
        cnn_captcha.predict('./images/234.png','./model/10_checkpoint.tar')
    cnn_captcha.predict(img,checkpoint) takes a path to an image file and a path to a checkpoint generated as illustrated above. This function would return the predicted answer as a string.
* note:
          
    Images to be predicted should have the same width and height of the training data. Depending on the difference between training data and images to be predicted, the performance can be quite bad actually. Fetching data that is generated from the source that you want to predict if high accuracy rate is what you need.
