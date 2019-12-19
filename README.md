# Event-Anomaly-Detection

Anomalies are rare and of varying nature and supervised learning approaches will fall short in terms of performance when dealing with such data. Auto-encoders however, are perfect for this situation, because they can be trained on normal parts and don't require annotated data. Once trained, we can give it a feature representation for a part and compare autoencoder output with input. The larger the difference, the more likely the input contains an anomaly.
*	We're submitting a .ipynb file with our project which is easy to run once you have the necessary dataset 'UCSD’s Anomaly Detection Dataset’, more importantly it’s UCSD ped1 folder.
*	Our file has section named CAE, Spatio-Temporal Stacked CAE, LSTM- Based Stacked CAE, Regularity Scores Comparison and Training Loss Comparison, the first 3 of which are further subdivided into ‘Training Phase’ and ‘Testing Phase’. Each of the latter mentioned modules can be run as standalone.
*	If you’re running the code on colab (and we advise you to do so), store the code file in the same folder as the data.
*	We’re using mxnet deep learning module because we wanted to experiment with some other library than TensorFlow, Keras, and Pytorch and mxnet is Apache’s alternative to them. To download mxnet (only for colab files) use:
``` python
!pip mxnet-cu100
```

***1.	CAE:***

**1.1 Training Phase:**

*	 We resized the original images into (1,100,100) form using the code snippet shown below as it was conducive to our experiment.
``` python
UCSD_FOLDER=os.path.join(DRIVE_MOUNT, 'My Drive', 'UCSD_Anomaly_Dataset.v1p2')
train_files = sorted(glob.glob(UCSD_FOLDER+ '/UCSDped1/Train/*/*'))
train_images = np.zeros((len(train_files),1,100,100))
for idx, filename in enumerate(train_files):
    im = Image.open(filename)
    im = im.resize((100,100))
    train_images[idx,0,:,:] = np.array(im, dtype=np.float32)/255.0
np.save(UCSD_FOLDER+ '/UCSD_Anomaly_Dataset.v1p2.npy',train_images)
```
* Once the images are saved, you can run the model’s ‘Training Phase’ till it’s end without any changes. Make sure the file is running on the gpu as it has the command ctx = gpu()which will try to access the gpu.
* Once the training is done, we are saving the model’s parameters for easy access during the testing especially, if you want to just run the testing part of the code. Any changes in the training parameters can be made in this cell before the training loop starts.
``` python
im_train = np.load(UCSD_FOLDER+ '/UCSD_Anomaly_Dataset.v1p2.npy')
batch_size= 32
dataset = gluon.data.ArrayDataset(mx.nd.array(im_train, dtype=np.float32))
dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch='rollover',shuffle=True)
ctx = gpu()
num_epochs = 50
model = ConvolutionalAutoencoder()
model.hybridize()
model.collect_params().initialize(mx.init.Xavier('gaussian'), ctx=ctx)
loss_function = gluon.loss.L2Loss()
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 1e-4, 'wd': 1e-5})
```
**1.2 Testing Phase:**

*	Herein, we are using the images from Test024 and Test024_gt files from UCSDped1, you can replace them with some other file with images and ground truth herein if you want. Following this we resize the test images and mask in the same manner as the training images and save them as a dataloader object.

``` python
test_file = sorted(glob.glob(UCSD_FOLDER +'/UCSDped1/Test/Test024/*'))
test_file_gt = sorted(glob.glob(UCSD_FOLDER +'/UCSDped1/Test/Test024_gt/*'))
a = np.zeros((len(test_file_gt),2,100,100))
for idx,filename in enumerate(test_file):
    im = Image.open(filename)
    im = im.resize((100,100))
    a[idx,0,:,:] = np.array(im, dtype=np.float32)/255.0

for idx,filename in enumerate(test_file_gt):
    im = Image.open(filename)
    im = im.resize((100,100))
    a[idx,1,:,:] = np.array(im, dtype=np.float32)/255.0

dataset = gluon.data.ArrayDataset(mx.nd.array(a, dtype=np.float32))
dataloader = gluon.data.DataLoader(dataset, batch_size=1)
```
*	Following are the 3 helper functions provided for test-time evaluation for CAE and were modified later to accommodate the Stacked models: 
``` python
def plot_regularity_score(model,dataloader):
  """
  Calculated regularity score per frame:
  Regularity Score = 1 - (e_t - min@t(e_t))/max@t(e_t)
  where e_t = sum over pixelwise l2 loss for each frame
  """
  e_t = []
  for image in dataloader:
    img = image[:,0,:,:].reshape(1,1,image.shape[-2],image.shape[-1])
    img = img.as_in_context(mx.gpu())
    output = model(img)
    output = (output.asnumpy().squeeze()*255).reshape(100*100,1)
    img = (img.asnumpy().squeeze()*255).reshape(100*100,1)
    e_xyt = np.linalg.norm(output-img,axis=1,ord=2)
    e_t.append(np.sum(e_xyt))
  e_t_min = min(e_t)
  e_t_max = max(e_t)
  reg_scores = []
  for i in range(len(e_t)):
    reg_scores.append(1 - ((e_t[i]-e_t_min)/e_t_max))
  return reg_scores

model =  ConvolutionalAutoencoder()
model.load_parameters(UCSD_FOLDER+ "/autoencoder_ucsd.params",ctx=ctx)
reg_scores_cae = plot_regularity_score(model,dataloader)
```
``` python
def plot_anomaly(img, output, diff, H, threshold, counter,UCSD_FOLDER):
  """
  Plots the images along the axis to show the input, output of the model,
  difference between the 2, and their predicted anomalies as red dots on
  the input image.
  """
    fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(10, 5))
    ax0.set_axis_off()
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax0.set_title('input image')
    ax1.set_title('reconstructed image')
    ax2.set_title('diff ')
    ax3.set_title('anomalies')
    ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest') 
    ax1.imshow(output, cmap=plt.cm.gray, interpolation='nearest')   
    ax2.imshow(diff, cmap=plt.cm.viridis, vmin=0, vmax=255, interpolation='nearest')  
    ax3.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    x,y = np.where(H > threshold)
    ax3.scatter(y,x,color='red',s=0.1) 
    plt.axis('off')
    fig.savefig(UCSD_FOLDER+'/images/' + str(counter) + '.png')
```
``` python
def model_evaluation(model,dataloader):
  loss_l2_per_frame = []
  threshold = 4*255
  counter = 0
  test_loss_metric = gluon.loss.SigmoidBCELoss()
  loss_per_frame = 0
  im_list = []
  i = 0
  for image in dataloader:
    counter = counter + 1
    img = image[:,0,:,:].reshape(1,1,image.shape[-2],image.shape[-1])
    mask = image[:,1,:,:].as_in_context(mx.gpu())
    img = img.as_in_context(mx.gpu())
    output = model(img)
    output = output.transpose((0,2,3,1))
    img = img.transpose((0,2,3,1))
    output = output.asnumpy()*255
    img = img.asnumpy()*255
    diff = np.abs(output-img) 
    tmp = diff[0,:,:,0]
    H = signal.convolve2d(tmp, np.ones((4,4)), mode='same')
    H_new = mx.nd.array(np.where(H>threshold,1,0).reshape((1,100,100)),ctx=gpu())
    loss = test_loss_metric(H_new, mask)
    loss_l2_per_frame.append(loss.asscalar())
    plot_anomaly(img[0,:,:,0], output[0,:,:,0], diff[0,:,:,0], H, threshold, counter,UCSD_FOLDER)

  print("Total loss per frame for anomalies predicted = ",sum(loss_l2_per_frame)/len(dataloader))
```
*	The commands needed to call the function(s) is contained within the same cells.
*	We’ve also provided the code to create the video file from the images stored by the last command of the ‘plot_anomaly’ function. This function is used as same by the following models. The same function is made available in the code as the last cell of ‘Testing Phase’ for the following models.

``` python
## Saving the output plots as video depicting anomalies ##
import cv2
out_im = sorted(glob.glob(UCSD_FOLDER+ '/images/*.png'))

img_array = []
for filename in out_im:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    # size = (width,height)
    img_array.append(img)

size = (360, 720)
_name = UCSD_FOLDER+'/vid' + '.mp4'
# self._cap = VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out_vid = cv2.VideoWriter(_name,fourcc,15,size)

for i in range(0,199):
  out_vid.write(img_array[i])

out_vid.release()

```
***2.	Spatio-Temporal Stacked CAE:***

**2.1 Training Phase:**
``` python
files = sorted(glob.glob(UCSD_FOLDER+'/UCSDped1/Train/*/*'))
train_images = np.zeros((int(len(files)/n), n, 227, 227))
i = 0
idx = 0
for filename in range(0, len(files)):
    im = Image.open(files[filename])
    im = im.resize((n,n))
    a[idx,i,:,:] = np.array(im, dtype=np.float32)/255.0
    i = i + 1
    if i >= n:
      idx = idx + 1
      i = 0
np.save(UCSD_FOLDER + '/stacked_cae.npy',train_images)
```
*	The code above, depicts the method to create the stacked training images array from original images resized to (n,227,227) for the Stacked CAE and LSTM Stacked CAE models. It should be run only once and then you can access the stacked images from the UCSD_FOLDER + '/stacked_cae.npy' which will be created  via this code snippet.
*	Same as before, you can run the rest of the code in the part without any extra efforts. Any changes in the training parameters can be made at:
``` python
ctx = gpu()
im_train = np.load(UCSD_FOLDER + '/stacked_cae.npy')
batch_size=32
dataset = gluon.data.ArrayDataset(mx.nd.array(im_train, dtype=np.float32))
dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch='rollover',shuffle=True)
num_epochs = 50
model = convSTAE()
model.hybridize()
model.collect_params().initialize(mx.init.Xavier('gaussian'), ctx=ctx)
loss_function = gluon.loss.L2Loss()
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 1e-4, 'wd': 1e-5})
```
**2.2 Testing Phase:**
*	The shape of the test images and their anomaly masks are also modified as per the requirements of the model. We are using separate data loaders for masks and images. 
``` python
model =  convSTAE()
model.load_parameters(UCSD_FOLDER +'/autoencoder_stacked_ucsd.params',ctx=ctx)
batch_size= 1
n=10 
test_file = sorted(glob.glob(UCSD_FOLDER+ '/UCSDped1/Test/Test024/*'))
test_file_gt = sorted(glob.glob(UCSD_FOLDER+'/UCSDped1/Test/Test024_gt/*'))
a = np.zeros((int(len(test_file)/n), n, 227, 227))
i = 0
idx = 0
for filename in range(0, len(test_file)):
    im = Image.open(test_file[filename])
    im = im.resize((227,227))
    a[idx,i,:,:] = np.array(im, dtype=np.float32)/255.0
    i = i + 1
    if i >= n:
      idx = idx + 1
      i = 0

b = np.zeros((int(len(test_file_gt)/n), n, 227, 227))
i = 0
idx = 0

for filename in range(0, len(test_file_gt)):
    im = Image.open(test_file_gt[filename])
    im = im.resize((227,227))
    b[idx,i,:,:] = np.array(im, dtype=np.float32)/255.0
    i = i + 1
    if i >= n:
      idx = idx + 1
      i = 0
## Test-time dataloaders for true images and their anomaly masks ##
dataset = gluon.data.ArrayDataset(mx.nd.array(a,ctx= ctx,dtype=np.float32))
dataloader = gluon.data.DataLoader(dataset, batch_size=1)
test_dataset = gluon.data.ArrayDataset(mx.nd.array(b,ctx= ctx, dtype=np.float32))
test_dataloader = gluon.data.DataLoader(dataset, batch_size=1)
```
* 'loss_compute' function is used to evaluate the model as we need to extract the stacked frames and process them individuals. It needs the model, test images dataloader, test masks dataloader and the UCSD folder path.
``` python
def loss_compute(output,image_gt,image,UCSD_FOLDER,counter):
  loss_l2_per_frame = []
  test_loss_metric = gluon.loss.SigmoidBCELoss(from_sigmoid=False)
  # there will be 10 chnannels rep each image flatten them out
  output = output.asnumpy().squeeze()*255
  image_gt= image_gt.asnumpy().squeeze()
  image= image.asnumpy().squeeze()*255
  threshold = 4*255
  for i in range(0,10):
    counter+=1
    im_out = output[i,:,:]
    im = image[i,:,:]
    diff = np.abs(im_out-im)
    H = signal.convolve2d(diff, np.ones((4,4)), mode='same')
    H_new = mx.nd.array(np.where(H>threshold,1,0).reshape((1,227,227)),ctx=gpu())
    mask =  mx.nd.array(image_gt[i,:,:].reshape((1,227,227)),ctx=gpu())
    loss_l2_per_frame.append(test_loss_metric(H_new,mask).asscalar())
    plot_anomaly(im, im_out,diff, H, threshold, counter,UCSD_FOLDER)
  return loss_l2_per_frame
```
*	But, to call the ‘plot_anomaly’ and ‘loss_compute’, we need to use the ’model_evaluation’ cell, which then automatically gives you the loss value and the anomaly images.
``` python
def model_evaluation(model,dataloader,test_dataloader,UCSD_FOLDER):
  loss = []
  im_list = []
  counter = 0
  for image,image_gt in zip(dataloader,test_dataloader):
    output = model(image)
    l = loss_compute(output,image_gt,image,UCSD_FOLDER,counter)
    counter+=10
    loss.extend(l)
  print("Total loss per frame for anomalies predicted = ",sum(loss)/len(loss))
## Call statement  for evaluation ##
model_evaluation(model,dataloader,test_dataloader,UCSD_FOLDER)
```

***3.	LSTM- Based Stacked CAE:***

**3.1 Training Phase:**
*	For LSTM-network we don’t need to modify the training images, as we can simply use the  stacked images from the ```UCSD_FOLDER + '/stacked_cae.npy' ``` we created for the previous model but as there is a chance one may wish to just observe the results for this model, I’ve added the image synthesis code in this part for LSTM net.
*	Same as in case of previous models, if you encounter the following cell you can use it to modify the network training parameters and can run the training loop without any further action required on your part.
``` python
batch_size=8
dataset = gluon.data.ArrayDataset(mx.nd.array(im_train, dtype=np.float32))
dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch='rollover',shuffle=True)
model = ConvLSTMAE()
ctx = gpu()
num_epochs = 50
model.hybridize()
model.collect_params().initialize(mx.init.Xavier(), ctx=mx.gpu())
loss_function = gluon.loss.L2Loss()
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 1e-4, 'wd': 1e-5})
```
**3.2 Testing Phase:**
*	Test part of the LSTM network is the same as the Stacked CAE except for the ‘states’ of the model’s temporal encoder being input to the ‘model_evaluation’ function as follows:
``` python
batch_size=8
dataset = gluon.data.ArrayDataset(mx.nd.array(im_train, dtype=np.float32))
dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, last_batch='rollover',shuffle=True)
model = ConvLSTMAE()
ctx = gpu()
num_epochs = 50
model.hybridize()
model.collect_params().initialize(mx.init.Xavier(), ctx=mx.gpu())
loss_function = gluon.loss.L2Loss()
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 1e-4, 'wd': 1e-5})
states = model.temporal_encoder.begin_state(batch_size=batch_size, ctx=ctx)
```
***4.	Regularity Scores Comparison:***
*	This section contains the code to generate comparison plots and the function to calculate regularity score on stacked test images. Running this, will provide you with the visual of the individual model scores to compare.
``` python
def plot_regularity_score_on_stacked_images(model,dataloader,states=None,lstm=False):
  """
  Calculated regularity score per frame:
  Regularity Score = 1 - (e_t - min@t(e_t))/max@t(e_t)
  where e_t = sum over pixelwise l2 loss for each frame
  """
  e_t = []
  for image in dataloader:
    img = image.as_in_context(gpu())
    if lstm:
      output ,_ = model(img,states)
    else:
      output = model(img)
    output = output.asnumpy().squeeze()*255
    img = img.asnumpy().squeeze()*255
    for i in range(output.shape[0]):
      a = output[i,:,:].reshape(227*227,1)
      b = img[i,:,:].reshape(227*227,1)
      e_xyt = np.linalg.norm(a-b,axis=1,ord=2)
      e_t.append(sum(e_xyt))
  e_t_min = min(e_t)
  e_t_max = max(e_t)
  reg_scores = []
  for i in range(len(e_t)):
    reg_scores.append(1 - ((e_t[i]-e_t_min)/e_t_max))
  return reg_scores

model_stcae = convSTAE()
model_stcae.load_parameters(UCSD_FOLDER +'/autoencoder_stacked_ucsd.params',ctx=ctx)
model_lstm =  ConvLSTMAE()
model_lstm.load_parameters(UCSD_FOLDER +'/autoencoder_lstm_ucsd.params',ctx=ctx)
reg_scores_stcae = plot_regularity_score_on_stacked_images(model_stcae,dataloader,lstm=False)
states = model_lstm.temporal_encoder.begin_state(batch_size=batch_size, ctx=ctx)
reg_scores_lstm = plot_regularity_score_on_stacked_images(model_lstm,dataloader,states,lstm=True)
## Plots
plt.plot(reg_scores_cae,color ='red')
plt.plot(reg_scores_stcae,color = 'green')
plt.plot(reg_scores_lstm,color='blue')
plt.xlabel("frame number")
plt.ylabel("regularity score")
plt.title( "Regularity Score per frame")
plt.legend(['CAE','STCAE', 'LSTM-STCAE'])
plt.show()
```
***5. Training Loss Comparison:***
*	In this section, we call on the files that store the model loss during training to plot them on a single pane for comparative analysis.
``` python
loss_train = np.load(UCSD_FOLDER+'/loss_train_cae.npy')
loss_train_stacked = np.load(UCSD_FOLDER+'/loss_train_stacked.npy')
loss_train_lstm = np.load(UCSD_FOLDER+'/loss_train_lstm.npy')
plt.plot(loss_train,'r')
plt.plot(loss_train_stacked,'g')
plt.plot(loss_train_lstm,'b')
plt.title('Training Loss vs Epochs')
plt.legend(['CAE','STCAE','LSTM-STCAE'])
plt.xlabel('epoch')
plt.ylabel('reconstruction loss')
plt.show()
```
