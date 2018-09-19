import os

os.popen("cd ~/tensorflow").read()
# os.popen("source ./bin/activate").read()
os.popen("tensorboard --logdir=~/PycharmProjects/TensorFlow/simGAN/graphs --port 6006").read()