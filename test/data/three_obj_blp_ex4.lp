\ Model multiobj
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  Set0: Priority=3 Weight=1 AbsTol=0 RelTol=0.01
   x[0] + x[1]
  Set1: Priority=2 Weight=1 AbsTol=0 RelTol=0.01
   x[0] + x[2]
  Set2: Priority=1 Weight=1 AbsTol=0 RelTol=0.01
   x[1] + x[3]
Subject To
  Budget: x[0] + x[1] + 2 x[2] + x[3] <= 2.5
Bounds
  x[0] >= 0
  x[1] >= 0
  x[2] >= 0
  x[3] >= 0
Binaries
  x[0]
  x[1]
End
