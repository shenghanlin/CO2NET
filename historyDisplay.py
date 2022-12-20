# Simplified unet for fault segmentation
# The original u-net architecture is more complicated than necessary 
# for our task of fault segmentation.
# We significanlty reduce the number of layers and features at each 
# layer to save GPU memory and computation but still preserve high 
# performace in fault segmentation.

import numpy as np 
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
def main():
    history=np.load('./check_mse_final29/history_mse2.npy',allow_pickle='TRUE').item()
    historytest=np.load('./check_mseRotationtest/history_mseRotationtest.npy',allow_pickle='TRUE').item()
    showHistory(history,historytest)


def showHistory(history,historytest):
  # list all data in history
  print(history.history.keys())
  #'val_loss', 'val_accuracy', 'val_metric_precision', 'val_metric_recall', 'val_metric_F1score', 
  # 'loss', 'accuracy', 'metric_precision', 'metric_recall', 'metric_F1score', 'lr'
  if history.history['accuracy']:
    fig = plt.figure(figsize=(10,6))
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(historytest.history['accuracy'])
    plt.plot(historytest.history['val_accuracy'])
    plt.title('Model accuracy',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train(Rotation)', 'test(Rotation)','train(Without Rotation)', 'test(Without Rotation)'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig("acc1.png")
#   plt.show()

  # summarize history for loss
  if history.history['loss']:
    fig = plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(historytest.history['loss'])
    plt.plot(historytest.history['val_loss'])
    plt.title('Model loss',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train(Rotation)', 'test(Rotation)','train(Without Rotation)', 'test(Without Rotation)'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig("loss2.png")

#   plt.show()

#   if history.history['metric_recall']:
#     fig = plt.figure(figsize=(10,6))
#     plt.plot(history.history['metric_recall'])
#     plt.plot(history.history['val_metric_recall'])
#     plt.title('Model metric recall',fontsize=20)
#     plt.ylabel('metric recall',fontsize=20)
#     plt.xlabel('Epoch',fontsize=20)
#     plt.legend(['train', 'test'], loc='center right',fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=18)
#     plt.tick_params(axis='both', which='minor', labelsize=18)
#     plt.savefig("recall.png")

# #   plt.show()
#   if history.history['metric_F1score']:
#     fig = plt.figure(figsize=(10,6))
#     plt.plot(history.history['metric_F1score'])
#     plt.plot(history.history['val_metric_F1score'])
#     plt.title('Model metric_F1score',fontsize=20)
#     plt.ylabel('metric_F1score',fontsize=20)
#     plt.xlabel('Epoch',fontsize=20)
#     plt.legend(['train', 'test'], loc='center right',fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=18)
#     plt.tick_params(axis='both', which='minor', labelsize=18)
#     plt.savefig("F1.png")
# #   plt.show()
#   if history.history['metric_precision']:
#     fig = plt.figure(figsize=(10,6))
#     plt.plot(history.history['metric_precision'])
#     plt.plot(history.history['val_metric_precision'])
#     plt.title('Model metric_precision',fontsize=20)
#     plt.ylabel('metric_precision',fontsize=20)
#     plt.xlabel('Epoch',fontsize=20)
#     plt.legend(['train', 'test'], loc='center right',fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=18)
#     plt.tick_params(axis='both', which='minor', labelsize=18)
#     plt.savefig("preci.png")

#   if history.history['mean_iou_keras']:
#     fig = plt.figure(figsize=(10,6))
#     plt.plot(history.history['mean_iou_keras'])
#     plt.plot(history.history['val_mean_iou_keras'])
#     plt.title('Model mean_iou_keras',fontsize=20)
#     plt.ylabel('mean_iou_keras',fontsize=20)
#     plt.xlabel('Epoch',fontsize=20)
#     plt.legend(['train', 'test'], loc='center right',fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=18)
#     plt.tick_params(axis='both', which='minor', labelsize=18)
#     plt.savefig("mean_iou_keras.png")

# #   plt.show()

if __name__ == '__main__':
    main()