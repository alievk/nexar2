# Nexar Challenge 2

First of all:
- assuming Nexar dataset is in $HOME/datasets/nexar and train images are in the 'train' subfolder, test images are in the 'test' subfolder
- git clone https://github.com/duburlan/nexar2
- cd nexar2
- nexar$ git clone https://github.com/duburlan/caffe -b nexar caffe-ssd-nexar
- nexar$ cd caffe-ssd-nexar
- follow the instructions in the README to build Caffe

To train the model:
- download the pretrained imagenet model http://cs.unc.edu/~wliu/projects/ParseNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel and put it in caffe-ssd-nexar/models/VGGNet
- caffe-ssd-nexar$ python data/nexar2/convert_xml.py --single-class --train 40000 --val 10000 --gen-annos
- caffe-ssd-nexar$ data/nexar2/create_data.sh
- caffe-ssd-nexar& python examples/ssd/ssd_nexar.py
- when training is complete, find the model in models/VGGNet/nexar2/SSD_600x600

Or download the models from: https://drive.google.com/open?id=0BxE3k9iDu5SVY0xBbjA4MXFsaTQ

Evaluate:
- cd ..
- nexar$ mkdir out
- nexar$ python ssd_eval.py --gpu 0 --model1 /path/to/VGG_nexar2_SSD_600x600_iter_60000.caffemodel --def1 /path/to/deploy.prototxt --model2 /path/to/VGG_nexar2_SSD_600x600_iter_70000.caffemodel --def2 /path/to/deploy.prototxt
- evaluation csv will be written to ./out

Note:
- --def1 and --def2 must be the same deploy.prototxt files
- you can run ssd_eval.py in parallel on multiple GPUs using --gpu option and --from A and --to B options, where A and B is the start and the end image number. this will produce file in ./out called test_A-B_.csv. then you have to concatenate these files into one (don't forget to remove the headers from the csv files when concatenating).
