# written by lachlan mcintosh o:qn 15- August 2017

import sys
from gurobipy import *

# cols is the number of variables
# rows is the number of constrinats

def dense_optimize(rows, cols, c, Q, A, sense, rhs, lb, ub, vtype,
                   solution):

  model = Model()

  # Add variables to model
  vars = []
  for j in range(cols):
    vars.append(model.addVar(lb=lb[j], ub=ub[j], vtype=vtype[j]))

  # Populate A matrix
  for i in range(rows):
    expr = LinExpr()
    for j in range(cols):
      if A[i][j] != 0:
        expr += A[i][j]*vars[j]
    model.addConstr(expr, sense[i], rhs[i])

  # Populate objective
  obj = QuadExpr()
  for i in range(cols):
    for j in range(cols):
      if Q[i][j] != 0:
        obj += Q[i][j]*vars[i]*vars[j]
  for j in range(cols):
    if c[j] != 0:
      obj += c[j]*vars[j]
  model.setObjective(obj)

  # Solve
  model.optimize()

  # Write model to a file
  #model.write('dense.lp')

  if model.status == GRB.Status.OPTIMAL:
    x = model.getAttr('x', vars)
    for i in range(cols):
      solution[i] = x[i]
    return True
  else:
    return False

W = [[0.3, 0.2],[0.7,0.8]]
X = [[1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1], [1, 2, 2, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1]]

import numpy as np
W = np.transpose(W)
x = np.dot(W,x)
Wit = [[0.5,0.5],[0.5,0.5]]
xmean = np.round(np.mean(x,0))
Xit = [xmean,xmean]

#ss vectorise a td array into a one d
import itertools
def vectorise(M):
  return(list(itertools.chain.from_iterable(M)))

for i in range(10):
  print("ROUND %d\n\n\n\n\n",i)

  rows = 2
  cols = 4
  Q = (np.kron(np.eye(2),np.dot(Xit,np.transpose(Xit)))/2).tolist()
  c = vectorise((-np.dot(x,np.transpose(Xit))).tolist())
  A = [[1,1,0,0],[0,0,1,1]]
  sense = [GRB.EQUAL]*rows
  rhs = [1,1]
  lb = [0]*cols
  ub = [1]*cols
  vtype = [GRB.CONTINUOUS]*cols
  sol = [0]*cols

  # Optimize

  success = dense_optimize(rows, cols, c, Q, A, sense, rhs, lb, ub, vtype, sol)

  Wit = np.reshape(sol,[2,2])
  xit = np.dot(Wit,Xit).tolist()

  print('x')
  print(xit)
  print(x)
  print('X')
  print(Xit)
  print(X)
  print('W')
  print(Wit)
  print(W)

  print("############# P2")

  # now transpose the problem!
  Q2 = (np.kron(np.eye(13),np.dot(np.transpose(Wit),Wit))/2).tolist()
  # a small amount of regularisation will go a long way. 
  Q2 = Q2 + np.eye(26)*0.01
  c2 = vectorise((-np.dot(np.transpose(x),Wit)).tolist())
  rows2 = 0 
  cols2 = len(c2)
  A2 = np.eye(rows2) 
  sense2 = [GRB.GREATER_EQUAL]*rows2
  lb2 = [0]*cols2
  ub2 = [5]*cols2
  vtype2 = [GRB.INTEGER]*cols2
  sol2 = [0]*cols2
  rhs2 = [0]*cols2
  # add the flow constraints...
  # Optimize

  success = dense_optimize(rows2, cols2, c2, Q2, A2, sense2, rhs2, lb2, ub2, vtype2, sol2)

  Xit = np.transpose(np.reshape(sol2,[13,2]))
  xit = np.dot(Wit,Xit).tolist()

  print('x')
  print(xit)
  print(x)
  print('X')
  print(Xit)
  print(X)
  print('W')
  print(Wit)
  print(W)


