Telegram bot for detecting objects in images


# Technology stack


# Installation
## Running a Docker container
To install, clone the repository with the command `git clone`, then run the image build with the command `docker build -t detect_bot:py3 .`. After building the image, run the container with the command `docker run -it --gpus all -v.:/app -v ../storage:/storage --name detect_bot detect_bot:py3 bash`.  
**Full list of commands:**
```
git clone https://github.com/Spir1N/detection_bot.git

docker build -t detect_bot:py3 .
docker run -it --gpus all -v .:/app -v ../storage:/storage --name detect_bot detect_bot:py3 bash
```
## Run the bot
Next, for a successful launch, you had to get a token (If you don't know how to do this, then here are the instructions: https://docs.expertflow.com/cx/4.5/how-to-get-telegram-bot-token ).  
Next, you need to create a *.secret* file in the *bot* directory and add your token to it as follows:
```
TELEGRAM_BOT_TOKEN=<yor_telegram_bot_token>
```
There are 3 ways to run a bot:
1) Run a ready-made YOLO model from Ultraceuticals
2) Run your own model (YOLO, Faster R-CNN, DETR)
3) PASCAL VOC training and further use of the model

### Using a ready-made model from Ultralytics
Using a ready-made Ultralytics model is the easiest and fastest way.  
To do this, you just need to change line 22 in the file *api/server.py * on `mdl = YOLO( "yolo11n.pt ")` (you can also select the model size by simply changing the letter *n* in the model name to *s*, *m*, *l*, *x*).  
Next, run the bot with the command:
``` 
python main.py 
```

In this version, only the selected YOLO model will work.

### Using your own model
To run your own version of the model:
1) Place your model (YOLO, DETR, Faster R-CNN) in the following path: *storage/models/(yolo, detr, rcnn)/model_name.pt*.  
2) Enter the name of your model (YOLO) in the file *api/server.py *, namely, replace the name with your own in line 22.  
3) Enter the name of the DETR model in the file *models/detr/detr.py *, line 31.  

Next, run the bot with the command:
``` 
python main.py 
```

In this case, only the models that were uploaded will work in the bot.

### Using models with training
This repository was originally designed to conduct an experiment comparing YOLO, Faster R-CNN and DETR models on the PASCAL VOC dataset and involves pre-training the models.   
- In order to run model training, you can build a Docker container separately for training each model in order to distribute calculations for each training session to different servers (to speed up the training of all models). The repository already has Dockerfiles for each model.  
- You can also start training models from a Docker container that has already been assembled at a previous stage.

To start model training, enter the following commands:
#### YOLO:
```
cd models/yolo/train
python train_yolo.py
``` 
#### Faster R-CNN:
```
cd models/rcnn/train
python train_faster_rcnn.py
```
#### DETR:
```
cd models/detr/train
python train_detr.py
```
After completing the training to launch the bot, go back to the root of the repository and enter the command:
``` 
python main.py 
```

In this case, only those models that have been trained will work in the bot.