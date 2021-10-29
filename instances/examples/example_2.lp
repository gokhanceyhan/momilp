\ Model multiobj
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  z1: Priority=1 Weight=1 AbsTol=0 RelTol=0.01
   x[1]
  z2: Priority=2 Weight=1 AbsTol=0 RelTol=0.01
   x[2]
  z3: Priority=3 Weight=1 AbsTol=0 RelTol=0.01
   4 x[3] + 3 x[4] + 2 x[5]
Subject To
  c1: x[1] - 1000 x[3] + 1000 x[4] + 1000 x[5] >= -999
  c2: x[1] + 1000 x[3] - 1000 x[4] - 1000 x[5] <= 1003
  c3: x[2] - 1000 x[3] + 1000 x[4] + 1000 x[5] >= -999
  c4: x[1] + x[2] + 1000 x[3] - 1000 x[4] - 1000 x[5] <= 1006
  c5: 2 x[1] + x[2] + 1000 x[3] - 1000 x[4] - 1000 x[5] <= 1008
  c6: x[1] - 1000 x[3] + 1000 x[4] - 1000 x[5] <= 1001.5
  c7: x[1] + 1000 x[3] - 1000 x[4] + 1000 x[5] >= - 998.5
  c8: x[2] - 1000 x[3] + 1000 x[4] - 1000 x[5] <= 1001.5
  c9: x[2] + 1000 x[3] - 1000 x[4] + 1000 x[5] >= - 998.5
  c10: x[1] + 1000 x[3] + 1000 x[4] - 1000 x[5] >= - 998
  c11: x[1] - 1000 x[3] - 1000 x[4] + 1000 x[5] <= 1005
  c12: x[1] + 3 x[2] - 1000 x[3] - 1000 x[4] + 1000 x[5] <= 1011
  c13: x[1] + x[2] - 1000 x[3] - 1000 x[4] - 1000 x[5] <= 3
  c14: x[1] - x[2] - 1000 x[3] - 1000 x[4] - 1000 x[5] <= 0
  c15: x[3] + x[4] + x[5] <= 1
Bounds
  x[1] >= 0
  x[2] >= 0
Binaries
  x[3]
  x[4]
  x[5]
End
