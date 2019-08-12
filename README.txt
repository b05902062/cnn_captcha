This is a easy to use pytorch captcha tool.
By Wang Zihmin 2019

* description:

There were a captcha generation tool and a model only in tensorflow at https://github.com/JackonYang/captcha-tensorflow. Furthermore, it hasn't been updated for a while, some of the code in it has been depreciated.
Thus, I wrote a pytorch captcha solver with more functionality and an easy to use interface. I borrowed the captcha generation tool from the above link and slightly modify it to generate data.

* gen_captcha.py usage:

 
    * generate captcha:
        * command line
            Please refer to the above link for more detailed usage.
            parameters -s -h -d are added to the original file. 
        
            e.g. python3 gen_captcha.py -n 10 -s 1000 --npi 5 -d -w 140 -h 45
            This would generate 10*1000 images of size height 45 * width 140 containing 5 digits.
        
* cnn_captcha.py usage: 

        '-d','--data',help='A path to the directory containing training data (images).')
        '-l','--loadcheck',help='filename of a checkpoint, relative to current working directory')
        images).')
        '-i','--image',help='A path to the image to precict.')
        '-e','--epoch',default=30,help='total number of epoch for either new model or resumed model. default to 30. e.g. -e 30 -l 15_checkpoint.tar would train this model for 15 more epoches.')
        '--convLayer',default=2,help='number of layer for convNet')
        '--convKernel',default=5,help='size of kernel for convNet.')
        '--fcLayer',default=3,help='number of layer for fcNet')
        '--pretrainedModel',help='load pretrained convolution Model, for captcha with many letters are hard to train directly. e.g. load a 2 digit checkpoint to train a 5 digit. the height and width of the images should be the same')
        '--fixConv',help='fix the parameters in convLayers.')
   
   * train model:
    
        * command line
            possible parameter -d -l -e --convLayer --convKernel --fcLayer --pretrainedModel --fixConv
            required parameter -d
            
            e.g. python3 ./cnn_captcha.py -d ./images/char-2-epoch-1 
            e.g. python3 ./cnn_captcha.py -d ./images/char-2-epoch-1 -l ./images/char-2-epoch-1/30_checkpoint.tar -e 60
            e.g. python3 cnn_captcha.py -d images/char-5-epoch-10/ --pretrainedModel ./images/char-2-epoch-10/30_checkpoint.tar

    * predict captcha:
        * command line
            possible parameter -p -i -l
            required parameter -p -i -l
    
    
            e.g. python3 ./cnn_captcha.py -p -i ./images/234.png -l ./model/10_checkpoint.tar
            
        * function call
        
            import cnn_captcha
            cnn_captcha.predict('./images/234.png','./model/10_checkpoint.tar')
