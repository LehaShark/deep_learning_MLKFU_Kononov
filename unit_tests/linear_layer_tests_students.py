import unittest
import numpy as np
from torch.nn import functional
from torch import Tensor
from torch.autograd import gradcheck

class TestMLP(unittest.TestCase):
    def test1(self):
        test_input = np.random.normal(size=(64, 28*28))
        fc1 = Linear()
        result = fc1(test_input)

        torch_result = functional.linear(Tensor(test_input), Tensor(fc1.weights).t())
        self.assertEqual(np.array_equal(result.shape, torch_result.numpy().shape), True,
                         (result.shape, torch_result.numpy().shape))
        self.assertEqual(np.allclose(result, torch_result.numpy(), atol=1e-6), True)
        # gradcheck

    def test_fc_size(self):
        """
        в цикле с разным размером полносвязного слоя проверить значения выхода полносвязного слоя и градиентов
        """
        pass

    def test_activation_function(self):
        """
        в цикле по разным функциям активации проверить значения выхода функции активации и градиентов
        """
        pass

    def test_nrof_layers(self):
        """
        в цикле по количеству слоев проверить -//-
        """
        pass




if __name__ == "__main__":
  unittest.main()