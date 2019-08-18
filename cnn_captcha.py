import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import argparse
import json
from torch.utils.data import Dataset, DataLoader
import logging
import skimage.io as io
import cv2

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOGGER = None
DATADIR=None
META=None

def getConsoleHandler():
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(FORMATTER)
	return consoleHandler

def getFileHandler(logFile,level):
	fileHandler = logging.FileHandler(logFile)
	fileHandler.setFormatter(FORMATTER)
	fileHandler.setLevel(level)
	return fileHandler

def getLogger(loggerName):
	if not os.path.exists(os.path.join(DATADIR,'log')):
		os.makedirs(os.path.join(DATADIR,'log'),0o0700)
	logger = logging.getLogger(loggerName)
	logger.setLevel(logging.DEBUG) # better to have too much log than not enough
	logger.addHandler(getConsoleHandler())
	logger.addHandler(getFileHandler(os.path.join(DATADIR,'log/info'),logging.INFO))
	logger.addHandler(getFileHandler(os.path.join(DATADIR,'log/debug'),logging.DEBUG))
	# with this pattern, it's rarely necessary to propagate the error up to parent
	logger.propagate = False
	return logger

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count if self.count != 0 else 0
'''
class my_captchaDataset(Dataset):

	def __init__(self,dataList,dataAns,batchSize,eval=None):

		self.data=dataList
		self.ans=dataAns
		self.batchSize=batchSize
		self.eval=eval
	def __len__(self):
		return len(self.data)
	def __iter__(self):
	
		if not self.eval:
			perm=torch.randperm(len(self))
		else:
			perm = torch.arange(len(self))

		for i in range((len(self)-1+self.batchSize)//self.batchSize):
			batchIdx=perm[i*self.batchSize:(i+1)*self.batchSize]
			imgList=[]
			ansList=[]
			for o in batchIdx:
				imgList.append(torch.from_numpy(io.imread(self.data[o])).type(torch.FloatTensor).unsqueeze(dim=0))
				ansList.append(self.ans[o].unsqueeze(dim=0))	
			imgList=torch.cat(imgList,0)
			ansList=torch.cat(ansList,0)
			#LOGGER.debug(f'{imgList},{ansList}')

			yield imgList,ansList
'''

class captchaDataset(Dataset):

	def __init__(self,dataList,dataAns):

		self.data=dataList
		self.ans=dataAns
	def __len__(self):
		return len(self.data)
	def __getitem__(self,idx):

		#image is a tensor
		image=cv2.imread(self.data[idx],0 if META['grayScale'] else 1)
		image=torch.from_numpy(image).type(torch.FloatTensor)
		if META['grayScale']:
			image=image.unsqueeze(2)


		#ans is a tensor of length meta['num_per_image']
		ans=self.ans[idx]

		return (image,ans)



def preprocess():

	with open(os.path.join(DATADIR,'meta.json')) as f:
	    meta = json.load(f)
	    f.close()

	LOGGER.info(f"image meta info {meta}.")
	
	#List is a list of string(file name) of size equals the number of training images.
	trainImgList=os.listdir(os.path.join(DATADIR,'train'))
	trainImgAns=trainImgList.copy()
	trainImgList=[os.path.join(DATADIR,"train",i) for i in trainImgList]

	#Ans becomes a list of tensor(answer) of size equals the number of training images*num_per_image.
	trainImgAns=[ans[:meta['num_per_image']] for ans in trainImgAns]
	trainImgAns=[ torch.tensor([meta['label_choices'].find(digit) for digit in ans]) for ans in trainImgAns]	
	testImgList=os.listdir(os.path.join(DATADIR,'test'))
	testImgAns=testImgList.copy()
	testImgList=[os.path.join(DATADIR,"test",i) for i in testImgList]

	testImgAns=[ans[:meta['num_per_image']] for ans in testImgAns]
	testImgAns=[ torch.tensor([meta['label_choices'].find(digit) for digit in ans]) for ans in testImgAns]	
	
	LOGGER.debug("{} {}".format(trainImgList[:3],trainImgAns[:3]))

	return (meta,trainImgList,trainImgAns,testImgList,testImgAns)

class cnnNet(nn.Module):
	def __init__(self):
		super(cnnNet, self).__init__()
		LOGGER.info(f"input image size {str(META['height'])}*{str(META['width'])} height * width")
		def layerSize(height,width):
			height=(height-META['convKernel'])+1 #conv2d refer to pytorch doc
			height=math.floor( ((height-2)/2) + 1) #pool refer to pytorch doc
			width=(width-META['convKernel'])+1
			width=math.floor( ((width-2)/2) + 1)
			return (height,width)

		self.linearH,self.linearW=META['height'],META['width']
		for i in range(META['convLayer']):
			(self.linearH,self.linearW) = layerSize(self.linearH,self.linearW)
		
		LOGGER.debug("image size after all conv2d height*width {}*{}".format(self.linearH,self.linearW))
		if self.linearH <= 0 or self.linearW <=0:
			LOGGER.info("convLayer too many or image too small")
			exit()
		self.convLayer=META['convLayer']
		self.convKernel=META['convKernel']
		self.cnnList=nn.ModuleList()
		for i in range(self.convLayer):
			#nn.Conv2d(in_channel,out_channel,kernel)
			if i==0 and not META['grayScale']:
				self.cnnList.append(nn.Conv2d(3,16,self.convKernel))
			elif i==0:
				self.cnnList.append(nn.Conv2d(1,16,self.convKernel))
			else:
				self.cnnList.append(nn.Conv2d(16,16,self.convKernel))

		#nn.MacPool2d(kernel,stride),fixed here.
		self.pool=nn.MaxPool2d(2,2)
	
	def forward(self,x):
		#x becomes batch*channel*height*width
		x=x.transpose(1,3).transpose(2,3)
		for i in range(self.convLayer):
			x = self.pool(F.relu(self.cnnList[i](x)))
		return x


class fcNet(nn.Module):
	def __init__(self):
		super(fcNet, self).__init__()
		
		def layerSize(height,width):
			height=(height-META['convKernel'])+1 #conv2d refer to pytorch doc
			height=math.floor( ((height-2)/2) + 1) #pool refer to pytorch doc
			width=(width-META['convKernel'])+1
			width=math.floor( ((width-2)/2) + 1)
			return (height,width)

		self.linearH,self.linearW=META['height'],META['width']
		for i in range(META['convLayer']):
			(self.linearH,self.linearW) = layerSize(self.linearH,self.linearW)
		self.fcLayer=META['fcLayer']
		self.fcList=nn.ModuleList()
		for i in range(self.fcLayer):
			if i==0 and self.fcLayer==1:
				self.fcList.append(nn.Linear(16*self.linearH*self.linearW,META['num_per_image']*META['label_size']))
			elif i==0:
				self.fcList.append(nn.Linear(16*self.linearH*self.linearW,128))
			elif i != (self.fcLayer-1):
				self.fcList.append(nn.Linear(128,128))
			else:
				self.fcList.append(nn.Linear(128,META['num_per_image']*META['label_size']))
	
	def forward(self,x):

		x = x.view(-1, 16 * self.linearH * self.linearW)
		for i in range(self.fcLayer-1):
			x = F.relu(self.fcList[i](x))
		x = self.fcList[self.fcLayer-1](x)#last layer without relu.
		#x is a 2d tensor of size batch*(numper_image*label_size)
		return x

class captchaNet(nn.Module):
	def __init__(self):
		super(captchaNet, self).__init__()

		self.cnnNet=cnnNet()
		self.fcNet=fcNet()
	def forward(self,x):
		x=self.cnnNet(x)
		x=self.fcNet(x)
		return x

def evaluate(output,ans):
	#output is a 2d-tensor of size batchSize*(num_per_image*label_size)

	batchSize=len(ans)
	#output becomes batchSize*num_per_image*label_size
	output=output.view(batchSize,META['num_per_image'],META['label_size'])
	pred=getAns(output)
	letterAcc= (ans==pred).type(torch.FloatTensor).mean()
	imageAcc= (ans==pred).min(dim=1).values.type(torch.FloatTensor).mean() #use min(1) for or

	#LOGGER.debug("pred {}, ans {}, letterAcc {}, imageAcc {}".format(pred,ans,letterAcc,imageAcc))

	#ouput becomes (batchSize*num_per_image) * label_size)
	output=output.view(-1,META['label_size'])
	ans=ans.view(batchSize*META['num_per_image'])
	y=torch.zeros(batchSize*META['num_per_image'],META['label_size']).cuda(ans.device)#use the same device as others.
	y[range(y.shape[0]),ans]=1
	return F.binary_cross_entropy_with_logits(output,y),letterAcc,imageAcc

def getAns(output):
	#return a tensor
	return output.argmax(2)
	
def train(data):

	
	trainLoader = DataLoader(captchaDataset(data[1],data[2]), batch_size=META['batchSize'],shuffle=True)
	testLoader = DataLoader(captchaDataset(data[3],data[4]), batch_size=META['batchSize'],shuffle=False)
	#trainLoader=my_captchaDataset(data[1],data[2],batchSize=16)
	#testLoader=my_captchaDataset(data[3],data[4],batchSize=16)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	LOGGER.info("using {}".format(device))

	net=captchaNet()
	net.to(device)
	
	start_epoch=0
	
	if META['printEveryBatch']>( (len(data[1])-1+META['batchSize'])//META['batchSize']   ):
		META['printEveryBatch']=( (len(data[1])-1+META['batchSize'])//META['batchSize'] )

	if META['loadCheck']:
		checkpoint = torch.load(META['loadCheck'])
		checkpoint['META']
		cnn_sd = checkpoint['cnnModel']
		fc_sd = checkpoint['fcModel']
		#optimizer_sd = checkpoint['optim']
		start_epoch= checkpoint['iteration']
		LOGGER.info(f"resume training at {start_epoch+1}th epoch.")
		if start_epoch>=META['epoch']:
			LOGGER.info("epoch number too smaller")
			exit()
		net.cnnNet.load_state_dict(cnn_sd)
		net.fcNet.load_state_dict(fc_sd)
		#optimizer.load_state_dict(optimizer_sd)
	if META['pretrainedModel']:
		checkpoint = torch.load(META['pretrainedModel'])
		cnn_sd = checkpoint['cnnModel']
		LOGGER.info(f"loading pretrained model at {META['pretrainedModel']}")
		net.cnnNet.load_state_dict(cnn_sd)
	if META['fixConv']:
		for p in net.cnnNet.parameters():
			p.requires_grad=False


	optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad ],lr=META['learnRate'],weight_decay=META['weightDecay'])
	LOGGER.info("using Adam optimizer"+"learning rate="+str(META['learnRate']))

	loss=0
	for o in range(start_epoch,META['epoch']):
		start_epoch+=1
		for i, batch in enumerate(trainLoader):


			net.train()
			#(batch*height*width,batch*num_per_img)
			batch=(batch[0].to(device),batch[1].to(device))

			#LOGGER.debug(f'cuda memory {torch.cuda.memory_allocated()}')
			output=net(batch[0])
			loss,_,_ =evaluate(output,batch[1])#calling batch[1] would get ans.
				

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if i % META['printEveryBatch']==META['printEveryBatch']-1:	
				net.eval()
				meterBCELoss=AverageMeter()
				meterImgAcc=AverageMeter()
				meterLetAcc=AverageMeter()
				#LOGGER.debug(f'cnn parametres {list(net.cnnNet.parameters())}')
				for batch in testLoader:
							
					batch=(batch[0].to(device),batch[1].to(device))
					output=net(batch[0])
					testEval=evaluate(output,batch[1])
					meterBCELoss.update(testEval[0].item(),batch[1].shape[0])
					meterLetAcc.update(testEval[1].item(),batch[1].shape[0])
					meterImgAcc.update(testEval[2].item(),batch[1].shape[0])
			
				LOGGER.info("epoch: {:03d}, {:05d}th batch in_loss: {:.3f}, out_loss: {:.3f}, outLetterAcc: {:.3f}, outImageAcc: {:.3f}".format(start_epoch,i+1,loss, meterBCELoss.avg,meterLetAcc.avg,meterImgAcc.avg))

	torch.save({"iteration":start_epoch,"cnnModel":net.cnnNet.state_dict(),'fcModel':net.fcNet.state_dict(),"optim":optimizer.state_dict(),"META":META,"letAcc":meterLetAcc.avg,"imgAcc":meterImgAcc.avg},os.path.join(DATADIR,"{}_{}.tar".format(start_epoch,"checkpoint")))
	LOGGER.info(f'model saved at {os.path.join(DATADIR,"{}_{}.tar".format(start_epoch,"checkpoint"))}')


def predict(imgFile,loadCheck):
	
	global DATADIR,META,LOGGER

	DATADIR = "./predict/"
	if not os.path.exists(DATADIR):
		os.makedirs(DATADIR,0o0700)
	LOGGER=getLogger("captchaLogger")
	
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#use cpu directly seem to be faster.
	device = torch.device("cpu")
	LOGGER.info("using {}".format(device))

	checkpoint = torch.load(loadCheck,map_location=device)
	META=checkpoint['META']	

	net=captchaNet()
	net.to(device)

	cnn_sd = checkpoint['cnnModel']
	fc_sd = checkpoint['fcModel']
	net.cnnNet.load_state_dict(cnn_sd)
	net.fcNet.load_state_dict(fc_sd)

	net.eval()

	#img=torch.from_numpy(io.imread(imgFile)).type(torch.FloatTensor).to(device).unsqueeze(dim=0)

	'''
	arbitrarily pass something to imgFile and fetch a image to store in imgFile here?
	probably comment the below if statement, too.
	'''

	img=cv2.imread(imgFile,0 if META['grayScale'] else 1)#1 for a color image, 0 for a grayScale one.
	if img is None:
		LOGGER.info("Error:image file not found")
		exit()	
	if img.shape[0]!=META['height'] or img.shape[1]!=META['width']:
		LOGGER.info("Error: images to be predicted should have the same width and height of the training data.")
		exit()

	img=torch.from_numpy(img).type(torch.FloatTensor).to(device).unsqueeze(dim=0)
	if META['grayScale']:
		img=img.unsqueeze(dim=3)
	ans=net(img).view(META['num_per_image'],META['label_size']).argmax(dim=1)
	ans=''.join(META['label_choices'][i] for i in list(ans))
	
	'''
	some postprocess using ans, which is a string,?
	'''

	LOGGER.info("Predicted captcha {}".format(ans))
	handlers=LOGGER.handlers[:]
	for i in handlers:
		i.close()
		LOGGER.removeHandler(i)
	return ans

def main():

	parser = argparse.ArgumentParser(description='simple captcha solver. All path should be passed relative to current working directory')

	
	parser.add_argument('-d','--data',help='A path to the directory containing training data (images).')
	parser.add_argument('-b','--batchSize',type=int,default=16,help='batch size')
	parser.add_argument('-l','--loadCheck',help='path to a checkpoint. -h -w --npi of gen_captcha and --convLayer --convKernel --fcLayer if specified explicitly, if you are using default then it will be ok, in this file should be the same as checkpoint, for we are loading both conv and fc')
	parser.add_argument('-e','--epoch',type=int,default=30,help='total number of epoch for either new model or resumed model. e.g. -e 30 -l 15_checkpoint.tar would train this model for 15 more epoches.')

	parser.add_argument('-p','--predict',action='store_true',help='Predict mode.The image to be predicted should have the same width and height of the training data. prediction mode ignore all parameters except -i and -l and itself.')
	parser.add_argument('-i','--image',help='A path to "an" image to precict. It accept only an image at a time now')

	
	parser.add_argument('--printEveryBatch',type=int,default=10,help='print loss every this number of unit.')
	parser.add_argument('--learnRate',type=int,default=0.001,help='learning rate for the optimizer')
	parser.add_argument('--weightDecay',type=int,default=0,help='L2 regulizer')
	parser.add_argument('--grayScale',action='store_true',help='Load in all training images as gray scale images. When using this mode all the following operation should and will be performed in this mode.')

	parser.add_argument('--convLayer',type=int,default=2,help='number of layers for convolution network. More layers may be able to capture a larger pattern, especially when the image size is big.')
	parser.add_argument('--convKernel',type=int,default=5,help='size of the kernel for all convolution network. Larger kernel may be able to capture a larger pattern in a image, especially when the image size is big.')
	parser.add_argument('--fcLayer',type=int,default=3,help='number of layers for fully connected network. It results in a bigger model.')
	parser.add_argument('--pretrainedModel',help='load pretrained convolution Model, for captcha with many letters are hard to train directly. e.g. load a 2 digit checkpoint to train a 5 digit model. -w -h of gen_captcha.py and --convLayer --convKernel if specified explicitly, if you are using default then it will be ok, should be the same as the old one, for we are loading a conv model.')
	parser.add_argument('--fixConv',action='store_true',help='fix the parameters in convLayers.')

	args = parser.parse_args()

	#check whether there is conflict in parameters.	
	if args.predict and (args.image is None or args.loadCheck is None):
		#predict
		parser.error("[-i --image] [-l --loadCheck] are required when predicting")
	elif (not args.predict and args.data is None):
		#training
		parser.error("[-d --data] is required when training.")
	elif args.convLayer <1:
		parser.error('convLayer has to be bigger of equal than 1')
	elif args.fcLayer <1:
		parser.error('fcLayer has to be bigger of equal than 1')
	elif args.pretrainedModel and args.loadCheck:
		parser.error('cannot loadCheck and load pretrainedModel at the same time')
	elif (not args.predict and args.loadCheck):
		flag=0
		checkpoint = torch.load(args.loadCheck)
		oldMETA=checkpoint['META']
		if oldMETA['convLayer']!=args.convLayer or oldMETA['convKernel']!=args.convKernel or oldMETA['fcLayer']!=args.fcLayer or oldMETA['grayScale']!=args.grayScale:
			flag=1

		with open(os.path.join(args.data,'meta.json')) as f:
			meta = json.load(f)
			f.close()
		
		if meta['height']!=oldMETA['height'] or meta['width']!=oldMETA['width'] or meta['num_per_image']!=oldMETA['num_per_image']:
			flag=1

		if flag:
			parser.error('when loading a checkpoint while training, --height --width --npi of gen_captcha and --convLayer --convKernel --fcLayer --grayScaleif specified explicitly, if you are using default then it will be ok, in this file should be the same as checkpoint, for we are loading both conv and fc')

	elif (not args.predict and args.pretrainedModel):
		flag=0
		checkpoint = torch.load(args.pretrainedModel)
		oldMETA=checkpoint['META']
		if oldMETA['convLayer']!=args.convLayer or oldMETA['convKernel']!=args.convKernel or oldMETA['grayScale']!=args.grayScale:
			flag=1

		with open(os.path.join(args.data,'meta.json')) as f:
			meta = json.load(f)
			f.close()
		
		if meta['height']!=oldMETA['height'] or meta['width']!=oldMETA['width']:
			flag=1

		if flag:
			parser.error('--width --height of gen_captcha.py and --convLayer --convKernel --grayScale if specified explicitly, if you are using default then it will be ok, should be the same as the old one, for we are loading a conv model.')


	#program actually starts here.
	global DATADIR,LOGGER,META

	DATADIR=args.data
	if args.predict:
		predict(args.image,args.loadCheck)
	else:

		LOGGER=getLogger("captchaLogger")

		LOGGER.info("process starts.")	
	
		LOGGER.info("start preprocessing...")
		data=preprocess()	
		LOGGER.info("preprocessing finished.")
	
		META={**data[0],**vars(args)}
		LOGGER.info("start training...")
		train(data)
		LOGGER.info("training finished")
		
		handlers=LOGGER.handlers[:]
		for i in handlers:
			i.close()
			LOGGER.removeHandler(i)


if __name__=="__main__":
	main()
