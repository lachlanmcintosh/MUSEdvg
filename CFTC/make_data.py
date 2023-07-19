#!/usr/local/bioinf/bin/python

import numpy as np

UPPER_BOUND = 3

OUTPUT = open('input.txt','w')
TRUTHW = open('truthW.txt','w')
TRUTHX = open('truthX.txt','w')

X = np.matrix(np.random.randint(UPPER_BOUND+1, size=(2, 4000)))

W = 200 * np.matrix([[0.02,0.98],[0.9, 0.1],[0.95,0.05]])

Z = np.transpose(W*X)
for row in Z:
  OUTPUT.write('\t'.join([str(s) for s in row.tolist()[0]]) + '\n')

X = np.transpose(X)
for row in X:
  TRUTHX.write('\t'.join([str(s) for s in row.tolist()[0]]) + '\n')

W = np.transpose(W)
for row in W:
  TRUTHW.write('\t'.join([str(s) for s in row.tolist()[0]]) + '\n')
