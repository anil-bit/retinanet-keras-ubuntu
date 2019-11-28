# implimentation of keras retinanet by Adibhatla venkat Anil

## Installation

1) Clone this repository.
2) Ensure numpy is installed using `pip install numpy --user`
3) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
4) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.
5) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.

## creating or preprocess for your custom dataset:

1) initially you  have to create 2 csv files.
2) 1st csv files contain the no of classes involved in training.
3) 2nd csv files contains name of the image file,location of object in image i.e coordinates of box,class name.
4) After creating 2 csv files place it in the downloaded keras-retinanet-master repository (i.e the main folder that you have extracted and unziped from github)
5) copy all the image that you want to train in the main folder i.e keras-retinanet-master repository

## How to start traininig?

In the terminal you need to type  pyhton keras_retinanet/bin/train.py csv /path/to/csv/file/containing/annotations.csv /path/to/csv/file/containing/classes.csv

#parametrs that you can adjust and try during training:
1)epochs: usually default is 50 epochs ,you can change it according to your requirment.example:pyhton keras_retinanet/bin/train.py --epochs 250 csv /path/to/csv/file/containing/annotations.csv /path/to/csv/file/containing/classes.csv(note:remember to set epochs after train.py and before csv,all the parameters that i am gonna mention below should place in same location after train.py and before csv)

2)'--snapshot':'Resume training from a snapshot.'

3)'--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True

4)'--weights',           help='Initialize the model with weights from a file.'
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights',     action='store_const', const=False)

 5)'--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str
 
 6)'--batch-size',       help='Size of the batches.', default=1, type=int
 
 7)'--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).'
 
 8)'--multi-gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0
 
 9)'--multi-gpu-force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true'
 

 10)'--epochs',          help='Number of epochs to train.', type=int, default=50
 
 11)'--steps',           help='Number of steps per epoch.', type=int, default=10000
 
 12)'--lr',              help='Learning rate.', type=float, default=1e-5
 
 13)'--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots'
 
 14)'--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs'
 
 15)'--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false'
 
 16)'--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false'
 
 17)'--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true'
 
 18)'--random-transform', help='Randomly transform image and annotations.', action='store_true'
 
 19)'--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800
 
 20)'--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333
 
 21)'--config',           help='Path to a configuration parameters .ini file.'
 
 22)'--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
 
 23)'--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true'

    # Fit generator arguments
 24)'--multiprocessing',  help='Use multiprocessing in fit_generator.', action='store_true'
 
 25)'--workers',          help='Number of generator workers.', type=int, default=1
 
 26)'--max-queue-size',   help='Queue length for multipnerator.', type=int, default=10)



### Projects using keras-retinanet
* [Improving Apple Detection and Counting Using RetinaNet](https://github.com/nikostsagk/Apple-detection). This work aims to investigate the apple detection problem through the deployment of the Keras RetinaNet.
* [Improving RetinaNet for CT Lesion Detection with Dense Masks from Weak RECIST Labels](https://arxiv.org/abs/1906.02283). Research project for detecting lesions in CT using keras-retinanet.
* [NudeNet](https://github.com/bedapudi6788/NudeNet). Project that focuses on detecting and censoring of nudity.
* [Individual tree-crown detection in RGB imagery using self-supervised deep learning neural networks](https://www.biorxiv.org/content/10.1101/532952v1). Research project focused on improving the performance of remotely sensed tree surveys.
* [ESRI Object Detection Challenge 2019](https://github.com/kunwar31/ESRI_Object_Detection). Winning implementation of the ESRI Object Detection Challenge 2019.
* [Lunar Rockfall Detector Project](https://ieeexplore.ieee.org/document/8587120). The aim of this project is to map lunar rockfalls on a global scale using the available > 1.6 million satellite images.
* [NATO Innovation Challenge](https://medium.com/data-from-the-trenches/object-detection-with-deep-learning-on-aerial-imagery-2465078db8a9). The winning team of the NATO Innovation Challenge used keras-retinanet to detect cars in aerial images ([COWC dataset](https://gdo152.llnl.gov/cowc/)).
* [Microsoft Research for Horovod on Azure](https://blogs.technet.microsoft.com/machinelearning/2018/06/20/how-to-do-distributed-deep-learning-for-object-detection-using-horovod-on-azure/). A research project by Microsoft, using keras-retinanet to distribute training over multiple GPUs using Horovod on Azure.
* [Anno-Mage](https://virajmavani.github.io/saiat/). A tool that helps you annotate images, using input from the keras-retinanet COCO model as suggestions.
* [Telenav.AI](https://github.com/Telenav/Telenav.AI/tree/master/retinanet). For the detection of traffic signs using keras-retinanet.
* [Towards Deep Placental Histology Phenotyping](https://github.com/Nellaker-group/TowardsDeepPhenotyping). This research project uses keras-retinanet for analysing the placenta at a cellular level.
* [4k video example](https://www.youtube.com/watch?v=KYueHEMGRos). This demo shows the use of keras-retinanet on a 4k input video.
* [boring-detector](https://github.com/lexfridman/boring-detector). I suppose not all projects need to solve life's biggest questions. This project detects the "The Boring Company" hats in videos.
* [comet.ml](https://towardsdatascience.com/how-i-monitor-and-track-my-machine-learning-experiments-from-anywhere-described-in-13-tweets-ec3d0870af99). Using keras-retinanet in combination with [comet.ml](https://comet.ml) to interactively inspect and compare experiments.
* [Weights and Biases](https://app.wandb.ai/syllogismos/keras-retinanet/reports?view=carey%2FObject%20Detection%20with%20RetinaNet). Trained keras-retinanet on coco dataset from beginning on resnet50 and resnet101 backends.
* [Google Open Images Challenge 2018 15th place solution](https://github.com/ZFTurbo/Keras-RetinaNet-for-Open-Images-Challenge-2018). Pretrained weights for keras-retinanet based on ResNet50, ResNet101 and ResNet152 trained on open images dataset. 
* [poke.AI](https://github.com/Raghav-B/poke.AI). An experimental AI that attempts to master the 3rd Generation Pokemon games. Using keras-retinanet for in-game mapping and localization.
* [retinanetjs](https://github.com/faustomorales/retinanetjs) A wrapper to run RetinaNet inference in the browser / Node.js. You can also take a look at the [example app](https://faustomorales.github.io/retinanetjs-example-app/).

If you have a project based on `keras-retinanet` and would like to have it published here, shoot me a message on Slack.

### Notes
* This repository requires Keras 2.3.0 or higher.
* This repository is [tested](https://github.com/fizyr/keras-retinanet/blob/master/.travis.yml) using OpenCV 3.4.
* This repository is [tested](https://github.com/fizyr/keras-retinanet/blob/master/.travis.yml) using Python 2.7 and 3.6.

Contributions to this project are welcome.

### Discussions
Feel free to join the `#keras-retinanet` [Keras Slack](https://keras-slack-autojoin.herokuapp.com/) channel for discussions and questions.

## FAQ
* **I get the warning `UserWarning: No training configuration found in save file: the model was not compiled. Compile it manually.`, should I be worried?** This warning can safely be ignored during inference.
* **I get the error `ValueError: not enough values to unpack (expected 3, got 2)` during inference, what to do?**. This is because you are using a train model to do inference. See https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model for more information.
* **How do I do transfer learning?** The easiest solution is to use the `--weights` argument when training. Keras will load models, even if the number of classes don't match (it will simply skip loading of weights when there is a mismatch). Run for example `retinanet-train --weights snapshots/some_coco_model.h5 pascal /path/to/pascal` to transfer weights from a COCO model to a PascalVOC training session. If your dataset is small, you can also use the `--freeze-backbone` argument to freeze the backbone layers.
* **How do I change the number / shape of the anchors?** The train tool allows to pass a configuration file, where the anchor parameters can be adjusted. Check [here](https://github.com/fizyr/keras-retinanet-test-data/blob/master/config/config.ini) for an example config file.
* **I get a loss of `0`, what is going on?** This mostly happens when none of the anchors "fit" on your objects, because they are most likely too small or elongated. You can verify this using the [debug](https://github.com/fizyr/keras-retinanet#debugging) tool.
* **I have an older model, can I use it after an update of keras-retinanet?** This depends on what has changed. If it is a change that doesn't affect the weights then you can "update" models by creating a new retinanet model, loading your old weights using `model.load_weights(weights_path, by_name=True)` and saving this model. If the change has been too significant, you should retrain your model (you can try to load in the weights from your old model when starting training, this might be a better starting position than ImageNet).
* **I get the error `ModuleNotFoundError: No module named 'keras_retinanet.utils.compute_overlap'`, how do I fix this?** Most likely you are running the code from the cloned repository. This is fine, but you need to compile some extensions for this to work (`python setup.py build_ext --inplace`).
* **How do I train on my own dataset?** The steps to train on your dataset are roughly as follows:
* 1. Prepare your dataset in the CSV format (a training and validation split is advised).
* 2. Check that your dataset is correct using `retinanet-debug`.
* 3. Train retinanet, preferably using the pretrained COCO weights (this gives a **far** better starting point, making training much quicker and accurate). You can optionally perform evaluation of your validation set during training to keep track of how well it performs (advised).
* 4. Convert your training model to an inference model.
* 5. Evaluate your inference model on your test or validation set.
* 6. Profit!
