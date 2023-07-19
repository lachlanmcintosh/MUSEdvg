
# Copyright 2017, Gurobi Optimization, Inc.

# This example formulates and solves the following simple QP model:
#  minimize
#      x^2 + x*y + y^2 + y*z + z^2 + 2 x
#  subject to
#      x + 2 y + 3 z >= 4
#      x +   y       >= 1
#
# It solves it once as a continuous model, and once as an integer model.

from gurobipy import *

W = [[0.3, 0.7],[0.2,0.8]]
X = [[1, 2, 2, 3, 2, 1, 1], [1, 1, 2, 2, 2, 2, 1 ]]
Y = [[1, 2, 2, 2, 1, 1, 1, 1, 0],[1, 1, 2, 2, 2, 1, 0, 0, 1]]
Z = [X[i] + Y[i] for i in range(len(X))]

locs_cns = [[ 
  ("loc1", 1),
  ("loc2", 1.3),
  ("loc3", 2),
  ("loc4", 2.3),
  ("loc5", 2),
  ("loc6", 1.3),
  ("loc7", 1)],[
  ("loc1", 1),
  ("loc2", 1.2),
  ("loc3", 2),
  ("loc4", 2.2),
  ("loc5", 2),
  ("loc6", 1.8), 
  ("loc7", 1)]
] 


bps_flows = [[
  ("loc1","loc1", 1),
  ("loc2","loc3", 1.3),
  ("loc3","loc4", 2),
  ("loc4","loc5", 2),
  ("loc5","loc6", 1.7),
  ("loc6","loc7", 1),
  ("loc5","loc2", 0.3),
  ("loc7","loc4", 0.3),
  ("loc7","loc3", 0.7)],[
  ("loc1","loc2", 1),
  ("loc2","loc3", 1.2),
  ("loc3","loc4", 2),
  ("loc4","loc5", 2),
  ("loc5","loc6", 1.8),
  ("loc6","loc7", 1),
  ("loc5","loc2", 0.2),
  ("loc7","loc4", 0.2),
  ("loc7","loc3", 0.8)]
]

# Create a new model
m = Model("qp")

# Create variables
locs_names = [locs_cns[i][j][0] for i in range(len(locs_cns)) for j in range(len(locs_cns[i]))]
CNS = m.addVars(locs_names,vtype=GRB.INTEGER,name="CNS")
flow_names = [bps_flows[i][j][0]+"_to_"+bps_flows[i][j][1] for i in range(len(bps_flows)) for j in range(len(bps_flows[i]))]
FLOWS = m.addVars(flow_names,vtype=GRB.INTEGER,name="FLOWS")

m.update()

def get_inouts(loc,dir=True):
  all = []
  for s in range(len(bps_flows)):
    for i in range(len(bps_flows[s])):
      if loc == bps_flows[s][i][int(dir)]:
        all.append(FLOWS[bps_flows[s][i][0]+"_to_"+bps_flows[s][i][1]])
  
  return(all)

for s in range(len(locs_cns)):
  for i in range(len(locs_cns[s])):
    outs = get_inouts(locs_cns[s][i][0],False)
    ins = get_inouts(locs_cns[s][i][0],True)
    name1="c_"+str(locs_cns[s][i][0])+"_out"
    name2="c_"+str(locs_cns[s][i][0])+"_in"
    m.addConstr(CNS[locs_cns[s][i][0]] == sum(outs),name1) 
    m.addConstr(CNS[locs_cns[s][i][0]] == sum(ins), name2)

# now to add the objective function
# can easily get the copy numbers and flows out,
# the objective function is them multiplied by the unknown ints -the known averages squared.

