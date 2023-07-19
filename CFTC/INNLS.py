# written by lachlan mcintosh between the 15th of August and the 2nd of September 2017

import sys
from gurobipy import *

EPSILON = 0.001
UPPER_BOUND = 3
TIME_LIMIT = 10 # this is the time limit in second to look for a solution in each round.
PLOIDY = 1.5

# rows is the number of 
def dense_optimize(rows, cols, c, Q, A, sense, rhs, lb, ub, vtype, solution):

  model = Model()

  # Add variables to model
  vars = []
  for j in range(cols):
    vars.append(model.addVar(lb=lb[j], ub=ub[j], vtype=vtype[j]))

  # Populate A matrix
  for i in range(np.shape(A)[0]):
    expr = LinExpr()
    for j in range(np.shape(A)[1]):
      if A[i][j] != 0:
        expr += A[i][j]*vars[j]
    model.addConstr(expr, sense[i], rhs[i])

  # Populate objective
  obj = QuadExpr()
  for i in range(np.shape(Q)[0]):
    for j in range(np.shape(Q)[1]):
      if Q[i][j] != 0:
        obj += Q[i][j]*vars[i]*vars[j]
  for j in range(cols):
    if c[j] != 0:
      obj += c[j]*vars[j]
  model.setObjective(obj,GRB.MINIMIZE)

  model.setParam('TimeLimit', TIME_LIMIT)

  # Solve
  model.optimize()
  #print(model)
  model.write("out.mst")
  model.write("debug.lp");
  model.write("out.sol")
 


  #print(model.status)
  #if model.status == GRB.Status.OPTIMAL:
  x = model.getAttr('x', vars)
  #print "TEMP" 
  #print x
  #print "TEMP"
  for i in range(cols):
    solution[i] = x[i]
  return True
  #else:
  #  return False


# vectorise as per the paper on NNMF
import itertools
import numpy as np
def vectorise(M):
  M = np.transpose(M)
  return(list(itertools.chain.from_iterable(M)))


# read in data:
input = open('input.txt','r')
Z = []
for line in input:
  bits = line.strip().split("\t")
  Z.append(bits)

input = open('truthW.txt','r')
W = []
for line in input:
  bits = line.strip().split("\t")
  W.append(bits)

input = open('truthX.txt','r')
X = []
for line in input:
  bits = line.strip().split("\t")
  X.append(bits)

Z = np.transpose(np.matrix(Z).astype(np.float))
W = np.matrix(W).astype(np.float)
X = np.transpose(np.matrix(X).astype(np.float))

print Z.shape
print W.shape
print Z.shape

import sys
num_normals = int(sys.argv[1])
num_tumours = int(sys.argv[2])

Z = np.matrix(Z)

tumour = np.matrix([[0.9, 0.1]])
normal = np.matrix([[0.1,0.9]])
Wit = normal
if num_normals > 1:
  for i in range(num_normals-1):
    Wit = np.concatenate((Wit,normal))
if num_tumours > 0:
  for i in range(num_tumours):
    Wit = np.concatenate((Wit,tumour))


#Wit = np.matrix.mean(Z)/(np.shape(Z)[0] * np.shape(Z)[1]) * Wit
Wit = np.matrix.mean(Z)/PLOIDY * Wit
Xit = np.matrix(np.concatenate((np.array([Z[1]]), np.array([Z[1]]))))

Z = np.array(Z)
Wit = np.array(Wit)
Xit = np.array(Xit)

i = 0
while True:
  i += 1
  print "ROUND" + str(i) + "\n\n"

  Xit = Xit_prev = np.array(Xit)
  Wit = Wit_prev = np.array(Wit)

  # now transpose the problem!
  Q2 = (np.kron(np.eye(np.shape(X)[1]),np.dot(np.transpose(Wit),Wit))/2)
  c2 = vectorise((-np.dot(np.transpose(Wit),Z)))
  rows2 = 0 
  cols2 = len(c2)
  A2 = np.eye(rows2) 
  sense2 = [GRB.GREATER_EQUAL]*rows2
  lb2 = [0]*cols2
  ub2 = [UPPER_BOUND]*cols2
  vtype2 = [GRB.INTEGER]*cols2
  sol2 = [vectorise(Xit)]*cols2
  rhs2 = [0]*cols2
  # Optimize

  success = dense_optimize(rows2, cols2, c2, Q2, A2, sense2, rhs2, lb2, ub2, vtype2, sol2)

  Xit = np.transpose(np.reshape(sol2,(np.shape(X)[1],np.shape(X)[0])))
  xit = np.dot(Wit,Xit)

  #print(sol2)
  print('X')
  print(Xit)
  print(X)
  print('W')
  print(Wit)
  print(W)

  print("############# P2")

  rows = 1
  Q = (np.kron(np.eye(np.shape(W)[1]),np.dot(Xit,np.transpose(Xit)))/2)
  c = vectorise((-np.dot(Xit,np.transpose(Z))))
  cols = len(c)
  r = 20
  #r2 = 8
  #A = [[-r2,1,1,-r]]
  A = [[0,1,0,-r]]
  sense = [GRB.GREATER_EQUAL]
  rhs = [0]
  lb = [0]*cols
  ub = [400]*cols
  vtype = [GRB.CONTINUOUS]*cols
  sol = [0]*cols

  # Optimize

  success = dense_optimize(rows, cols, c, Q, A, sense, rhs, lb, ub, vtype, sol)

  Wit = np.reshape(sol,(np.shape(W)[1],np.shape(W)[0]))
  Xit = np.array(Xit)

  print('X')
  print(Xit)
  print(X)
  print('W')
  print(Wit)
  print(W)



  if sum(sum(np.square(Xit-Xit_prev))) + sum(sum(np.square(Wit-Wit_prev))) < EPSILON:
    print "\n"+str(i) + " iterations"
    break 
