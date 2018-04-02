import numpy as np
from numpy import random

BIT_WIDTH = 20
NUM_CLASS = 4

class FizzBuzzDataSet(object):
    TOTAL_DATA = 0
    CURRENT_I = 0

    def __init__(self, size):
        if size < 0:
            size = 0
        self.TOTAL_DATA = size
        self.CURRENT_I = 0

    @staticmethod
    def _encode(num):
        return [num >> d & 1 for d in range(BIT_WIDTH)]

    @staticmethod
    def _encode_label(num):
        if num % 15 == 0:
            return 0
        elif num % 5 == 0:
            return 1
        elif num % 3 == 0:
            return 2
        else:
            return 3

    def _labels(self, nums):
        result_label = []
        for i in range(len(nums)):
            label_y = self._encode_label(nums[i])
            result_label.append(label_y)
        return np.array(result_label)

    def _numbers(self, size):
        """
        Assume the numbers for fizzbuzz won't be needing more than 10 bits < 1024
        """
        result_data = []
        result_num = []
        for i in range(size):
            num = random.randint(0, (1<<BIT_WIDTH)-1)
            result_num.append(num)
            encoded = self._encode(num)
            result_data.append(encoded)
        return np.array(result_data), result_num

    def next_batch(self, batch_size):
        """
        Returns:
            data
            label
        """
        size = batch_size
        input_x, real_nums = self._numbers(size)
        label_y = self._labels(real_nums)
        return input_x, label_y

    @property
    def num_examples(self):
        return self.TOTAL_DATA
