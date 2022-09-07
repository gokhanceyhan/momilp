\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  z_0: Priority=2 Weight=1 AbsTol=0 RelTol=0
   - 5 y[0] + 181 y[1] + 113 y[2] - 64 y[3] + 79 y[4]
  z_1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   10 x[0] + 9 x[1] + 5 x[2] + 5 x[3] - 2 x[4] - 74 y[0] + 156 y[1]
   - 117 y[2] + 88 y[3] - 57 y[4]
  z_2: Priority=0 Weight=1 AbsTol=0 RelTol=0
   6 x[0] - x[1] - 10 x[2] - 8 x[3] - x[4] + 79 y[0] - 45 y[1] - 34 y[2]
   + 66 y[3] + 39 y[4]
Subject To
 con_0: 4 x[0] + 10 x[1] + 11 x[2] + 7 x[3] + 8 x[4] + 16 y[0] <= 73
 con_1: 10 x[0] + 4 x[1] + 14 x[2] - x[3] + 15 x[4] + 2 y[1] <= 75
 con_2: 11 x[1] + 6 x[2] + 12 x[3] + 5 x[4] + 3 y[2] <= 57
 con_3: 17 x[0] + 19 x[1] + 4 x[2] + 17 x[3] + 19 x[4] + 16 y[3] <= 78
 con_4: 10 x[0] + 9 x[1] + 13 x[2] + 17 x[3] + 3 x[4] + 10 y[4] <= 60
 con_5: 8 x[0] + 16 x[1] - x[2] + 12 x[3] + 8 x[4] <= 96
 con_6: 8 x[0] + 6 x[1] - x[3] + 16 x[4] <= 82
 con_7: 7 x[0] + 12 x[1] + 18 x[2] + 14 x[3] + 9 x[4] <= 74
 con_8: 7 x[0] + 6 x[1] + 2 x[2] + 5 x[3] + 20 x[4] <= 73
 con_9: y[0] + y[1] + y[2] + y[3] + y[4] <= 1.666666666666667
Bounds
Binaries
 y[0] y[1] y[2] y[3] y[4]
End
