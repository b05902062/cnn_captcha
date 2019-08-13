This is a easy to use captcha tool in pytorch.
By Wang Zihmin 2019

* description:

There was a captcha generation tool at https://github.com/JackonYang/captcha-tensorflow. However, the model was only in tensorflow and it hasn't been updated for a while, some of the code in it has been depreciated.
Thus, I wrote a pytorch captcha solver. It has an easy to use interface for naive user and some functionalities to play with for more advanced use. I borrowed the captcha generation tool from the above link and slightly modify it to accommodate more needs.

* gen_captcha.py usage:

 
    * generate captcha:


	'-s',help='number of samples per epoch.'
	'--height',help='number of pixels in height.'
	'--width',help='number of pixels in width.'


        * command line
            Please refer to the above link for more detailed usage.
            parameters -s -height -width are added to the original file. 
        
            e.g. python3 gen_captcha.py -n 10 -s 1000 --npi=5 -d --width 140 --height 45
            This would generate 10*1000 images of size height 45 * width 140 containing 5 digits.
        

* cnn_captcha.py usage: 

	'-d','--data',help='A path to the directory containing training data (images).'
	'-b','--batchSize',default=16,help='batch size'
	'-l','--loadCheck',help='path to a checkpoint. -h -w --npi of gen_captcha and --convLayer --convKernel --fcLayer if specified explicitly, if you are using default then it will be ok, in this file should be the same as checkpoint, for we are loading both conv and fc'
	'-e','--epoch',default=30,help='total number of epoch for either new model or resumed model. e.g. -e 30 -l 15_checkpoint.tar would train this model for 15 more epoches.'

	'-p','--predict',help='Predict mode.'
	'-i','--image',help='A path to the image to precict.'

	'--printEveryBatch',default=800,help='print loss every this number of unit.'
	'--learnRate',default=0.001,help='learning rate for the optimizer'
	'--convLayer',default=2,help='number of layer for convNet'
	'--convKernel',default=5,help='size of kernel for convNet.'
	'--fcLayer',default=3,help='number of layer for fcNet'
	'--pretrainedModel',help='load pretrained convolution Model, for captcha with many letters are hard to train directly. e.g. load a 2 digit checkpoint to train a 5 digit model. -w -h of gen_captcha.py and --convLayer --convKernel if specified explicitly, if you are using default then it will be ok, should be the same as the old one, for we are loading a conv model.'
	'--fixConv',help='fix the parameters in convLayers.'
   
   * train model:
    
        * command line
            easy usage:
            
            possible parameter -d -l -e -b
            required parameter -d

            
            e.g. python3 ./cnn_captcha.py -d ./images/char-2-epoch-1 
            e.g. python3 ./cnn_captcha.py -d ./images/char-2-epoch-1 -l ./images/char-2-epoch-1/30_checkpoint.tar -e 60 -b 16

            more advanced usage:

            possible parameter -d -l -e --printEveryBatch --learnRate --convLayer --convKernel --fcLayer --pretrainedModel --fixConv
            required parameter -d

            e.g. python3 cnn_captcha.py -d images/char-5-epoch-10/ --convLayer 3 --convKernel 7 --fcLayer 4
            e.g. python3 cnn_captcha.py -d images/char-5-epoch-10/ --learnRate 0.0001 -b 16 --printEveryBatch 1 
            e.g. python3 cnn_captcha.py -d images/char-5-epoch-10/ --pretrainedModel ./images/char-2-epoch-10/30_checkpoint.tar --fixConv -e 15 --fcLayer 4

    * predict captcha:
            possible parameter -p -i -l
            required parameter -p -i -l
    
    
            e.g. python3 ./cnn_captcha.py -p -i ./images/234.png -l ./model/10_checkpoint.tar
            
        * function call
        
            import cnn_captcha
            cnn_captcha.predict('./images/234.png','./model/10_checkpoint.tar')
