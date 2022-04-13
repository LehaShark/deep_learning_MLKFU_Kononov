import numpy as np

class PhaseKeysDict(dict):
    def __init__(self, train: str = 'train', valid: str = 'valid', test: str = 'test'):
        super().__init__(train=train, valid=valid, test=test)


class Epoch(object):
    def __init__(self, model, criterion, optimizer, writer, dataloaders: dict, phase_keys: PhaseKeysDict = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = writer
        self.dataloaders = dataloaders
        self._phase_keys = phase_keys if phase_keys is not None else PhaseKeysDict()
        self._global_step = {v: 0 for v in phase_keys.values()}

    def train(self, i_epoch, show_each: int = 1, debug: bool = True):
        self.model.train()
        self._step(self._phase_keys['train'], i_epoch, show_each, debug)

    def validation(self, i_epoch, show_each: int = 1, debug: bool = True):
        self.model.eval()
        self._step(self._phase_keys['valid'], i_epoch, show_each, debug)

    def test(self, show_each: int = 1, debug: bool = True):
        self.model.eval()
        self._step(self._phase_keys['test'], 0, show_each, debug)

    def _step(self, phase, i_epoch, show_each, debug):
        i, ovr_loss, ovr_accuracy = 0, 0, 0
        for i, (images, labels) in enumerate(self.dataloaders[phase].batch_generator()):
            images = images.reshape(images.shape[0], -1)
            output = self.model(images)
            loss = self.criterion(output, labels)
            accuracy = np.mean(np.argmax(output.data, axis=1) == labels)
            if i % show_each == 0:
                print('epoch: ', i_epoch, ' iter: ', str(i), ' loss: ', loss.data, ' accuracy: ', accuracy)

            self.writer.add_scalar(f'{phase}/loss', loss.data, self._global_step[phase])
            self.writer.add_scalar(f'{phase}/accuracy', accuracy, self._global_step[phase])
            self._global_step[phase] += i
            ovr_loss += loss.data
            ovr_accuracy += accuracy

            if phase is self._phase_keys['train']:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if phase is not self._phase_keys['test']:
            self.writer.add_scalar(f'{phase}/epoch_loss', ovr_loss / i, i_epoch)
            self.writer.add_scalar(f'{phase}/epoch_accuracy', ovr_accuracy / i, i_epoch)

    def __repr__(self) -> str:
        return '{}(\n\tmodel={},\n\tcriterion={},\n\toptimizer={})'.format(
            type(self).__name__, str(self.model).replace('\n\t', ''), self.criterion, self.optimizer
        )

