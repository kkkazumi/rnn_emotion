#!/usr/bin/env python
# -*- coding:utf-8 -*-
 
import numpy as np
import math
import random
  
random.seed(0)
   
class DataMaker(object):
   
    def __init__(self, steps_per_cycle, number_of_cycles):
        self.steps_per_cycle = steps_per_cycle
        self.number_of_cycles = number_of_cycles
                        
    def make(self):
        return np.array([math.sin(i * 2 * math.pi/self.steps_per_cycle) for i in range(self.steps_per_cycle)] * self.number_of_cycles)

    def read(self, data_str):
        data = np.loadtxt(data_str,delimiter=",")
        return 0.1*data

    def make_mini_batch(self, data, mini_batch_size, length_of_sequence):
        sequences = np.ndarray((mini_batch_size, length_of_sequence), dtype=np.float32)
        print("make_data/",data)
        for i in range(mini_batch_size):
            index = random.randint(0, len(data) - length_of_sequence)
            sequences[i] = data[index:index+length_of_sequence]
        return sequences
