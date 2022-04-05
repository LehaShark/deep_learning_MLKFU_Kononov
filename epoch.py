import numpy as np


class Epoch(object):
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def step(self, dataloader, is_train, show_each=1, i_epoch=None):
        for i, (images, labels) in enumerate(dataloader.batch_generator()):
            images = images.reshape(images.shape[0], -1)
            output = self.model(images)
            loss = self.criterion(output, labels)

            if i % show_each == 0:
                print(
                      'iter: ' + str(i) + ' loss: ', loss.data,
                      'accuracy: ', np.mean(np.argmax(output.data, axis=1) == labels))

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

