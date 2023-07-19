#!/usr/local/bioinf/bin/python

import numpy as np
input = open('input.txt','r')
array = []
for line in input:
  bits = line.strip().split("\t")
  array.append(bits)

array = np.matrix(np.transpose(array))
print(array)

