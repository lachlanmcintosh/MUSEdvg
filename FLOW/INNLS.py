# written by lachlan mcintosh between the 15th of August and the 2nd of September 2017

import sys
from gurobipy import *

def dense_optimize(rows, cols, c, Q, A, sense, rhs, lb, ub, vtype, solution,eq_constr,S,size):

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
  
  if S>0: 
    for i in range(len(eq_constr)):
      for s in range(S):
        if len(eq_constr[i][0])>0:
          model.addConstr(sum([vars[lookup[x+s*size]] for x in eq_constr[i][0]]),GRB.EQUAL, vars[lookup[eq_constr[i][1]+s*size]],str(eq_constr[i][1])+"_lhs_"+str(s))
        if len(eq_constr[i][2])>0:
          model.addConstr(sum([vars[lookup[x+s*size]] for x in eq_constr[i][2]]),GRB.EQUAL,vars[lookup[eq_constr[i][1]+s*size]],str(eq_constr[i][1])+"_rhs_"+str(s))
  

  # Populate objective
  obj = QuadExpr()
  for i in range(cols):
    for j in range(cols):
      if Q[i][j] != 0:
        obj += Q[i][j]*vars[i]*vars[j]
  for j in range(cols):
    if c[j] != 0:
      obj += c[j]*vars[j]
  model.setObjective(obj,GRB.MINIMIZE)

  # Solve
  model.optimize()
  #print(model)
  model.write("out.mst")
  model.write("debug.lp");
  model.write("out.sol")

  #for v in model.getVars():
  #  if v.x != 0:
  #    print('%s %g' % (v.varName, v.x))

  #print(model.status)
  if model.status == GRB.Status.OPTIMAL:
    x = model.getAttr('x', vars)
    for i in range(cols):
      solution[i] = x[i]
      #print(solution[i])
    return True
  else:
    return False

W = [[0.3, 0.7],[0.2,0.8]]
X = [[1, 2, 2, 3, 2, 1, 1], [1, 1, 2, 2, 2, 2, 1 ]]
Y = [[1, 2, 2, 2, 1, 1, 1, 1, 0],[1, 1, 2, 2, 2, 1, 0, 0, 1]]

Z = [X[i] + Y[i] +[1,1-i,i] for i in range(len(X))]

# vectorise as per the paper on NNMF
import itertools
import numpy as np
def vectorise(M):
  M = np.transpose(M)
  return(list(itertools.chain.from_iterable(M)))

# some hard coded constraints - in general what will we use? what if one of these constraints is missed?
eq_constr = [[[16],0,[7]],
	[[7,13],1,[8]],
	[[8,15],2,[9]],
	[[9,14],3,[10,17]],
	[[10],4,[11,13]],
	[[11],5,[12,18]],
	[[12],6,[14,15]]]

lookup = vectorise(np.reshape(range(len(Z[0])*len(Z)),(len(Z[0]),len(Z)))) 

print("check that the truth satisfies the constraints:")
for i in range(2):
  for c in eq_constr:
    a=sum([Z[i][x] for x in c[0]]) == Z[i][c[1]]
    b=sum([Z[i][x] for x in c[2]]) == Z[i][c[1]]
    if(a!= True or b != True):
      print(a)
      print(b)
      print(sum([Z[i][x] for x in c[0]]) - Z[i][c[1]])
      print(sum([Z[i][x] for x in c[2]]) - Z[i][c[1]])
      print(Z)
      print(c)

z = np.dot(W,Z)

print("check that the mixture satisfies the constraints:")
for i in range(2):
  for c in eq_constr:
    a=sum([z[i][x] for x in c[0]]) == z[i][c[1]]
    b=sum([z[i][x] for x in c[2]]) == z[i][c[1]]
    if(a != True or b != True):
      print(a)
      print(b)
      print(sum([z[i][x] for x in c[0]]) - z[i][c[1]])
      print(sum([z[i][x] for x in c[2]]) - z[i][c[1]])
      print(z)
      print(c)


import random
for i in range(len(z)):
  for j in range(len(z[0])):
    z[i][j] += (random.random()-0.5)

Wit = [[0.25,0.75],[0.9,0.1]]
zmean = np.round(np.mean(z,0))
Zit = [zmean,zmean]

Z = np.array(Z)
Zit = np.array(Zit)



for i in range(10):
  print("ROUND %d\n\n\n\n\n",i)

  rows = 2
  cols = 4
  Q = (np.kron(np.eye(len(W)),np.dot(Zit,np.transpose(Zit)))/2)
  c = vectorise((-np.dot(Zit,np.transpose(z))))
  A = [[1,1,0,0],[0,0,1,1]]
  sense = [GRB.EQUAL]*rows
  rhs = [1,1]
  lb = [0]*cols
  ub = [1]*cols
  vtype = [GRB.CONTINUOUS]*cols
  sol = [0]*cols

  # Optimize

  success = dense_optimize(rows, cols, c, Q, A, sense, rhs, lb, ub, vtype, sol,[],0,0)

  Wit = np.reshape(sol,(len(W),len(W[0])))
  zit = np.dot(Wit,Zit)

  Zit = np.array(Zit)

  print("check")
  for i in range(2):
    for c in eq_constr:
      a=sum([Zit[i][x] for x in c[0]]) == Zit[i][c[1]]
      b=sum([Zit[i][x] for x in c[2]]) == Zit[i][c[1]]
      if(a!= True or b != True):
        print(a)
        print(b)
        print(sum([Zit[i][x] for x in c[0]]) - Zit[i][c[1]])
        print(sum([Zit[i][x] for x in c[2]]) - Zit[i][c[1]])
        print(Zit)
        print(c)

  print('z')
  print(zit)
  print(z)
  print('Z')
  print(Zit)
  print(Z)
  print('W')
  print(Wit)
  print(W)
  print(sum(sum(Z-Zit)))

  print("############# P2")

  # now transpose the problem!
  Q2 = (np.kron(np.eye(len(Z[0])),np.dot(np.transpose(Wit),Wit))/2)
  Q2 = Q2 + np.eye(len(Q2))*0.001
  c2 = vectorise((-np.dot(np.transpose(Wit),z)))
  rows2 = 0 
  cols2 = len(c2)
  A2 = np.eye(rows2) 
  sense2 = [GRB.GREATER_EQUAL]*rows2
  lb2 = [0]*cols2
  ub2 = [5]*cols2
  vtype2 = [GRB.INTEGER]*cols2
  sol2 = [vectorise(Zit)]*cols2
  rhs2 = [0]*cols2
  # add the flow constraints...
  # Optimize

  success = dense_optimize(rows2, cols2, c2, Q2, A2, sense2, rhs2, lb2, ub2, vtype2, sol2,eq_constr,len(Z),len(Z[0]))

  Zit = np.transpose(np.reshape(sol2,(len(Z[0]),len(Z))))
  zit = np.dot(Wit,Zit)

  print("constraint check")
  for i in range(2):
    for c in eq_constr:
      a=sum([Zit[i][x] for x in c[0]]) == Zit[i][c[1]]
      b=sum([Zit[i][x] for x in c[2]]) == Zit[i][c[1]]
      if(a!= True or b != True):
        print(a)
        print(b)
        print(sum([Zit[i][x] for x in c[0]]) - Zit[i][c[1]])
        print(sum([Zit[i][x] for x in c[2]]) - Zit[i][c[1]])
        print(Zit)
        print(c)

  print('z')
  print(zit)
  print(z)
  print('Z')
  print(Zit)
  print(Z)
  print('W')
  print(Wit)
  print(W)
  print(sum(sum(Z-Zit)))


