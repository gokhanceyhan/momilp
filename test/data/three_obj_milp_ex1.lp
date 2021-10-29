\ Model multiobj
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  Set0: Priority=3 Weight=1 AbsTol=0 RelTol=0.01
   x[0] + x[1] + x[2] + x[3] + x[4] + x[5]
  Set1: Priority=2 Weight=1 AbsTol=0 RelTol=0.01
   x[0] + x[1] + x[2] + x[3] + x[4] + x[5]
  Set2: Priority=1 Weight=1 AbsTol=0 RelTol=0.01
   x[0] + x[1] + x[2] + x[3] + x[4] + x[5]
Subject To
  Budget: 7 x[0] + 9 x[1] + 3 x[2] + 4 x[3] + 5 x[4] + 6 x[5] <= 50
Bounds
  x[0] >= 1
  x[1] >= 1
  x[2] >= 1
  x[3] >= 1
  x[4] >= 1
  x[5] >= 1
Generals
  x[0]
  x[1]
  x[2]
  x[3]
  x[4]
  x[5]
End
