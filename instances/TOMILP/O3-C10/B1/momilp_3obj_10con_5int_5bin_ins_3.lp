\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  z_0: Priority=2 Weight=1 AbsTol=0 RelTol=0
   - 166 y[0] + 46 y[1] - 181 y[2] + 124 y[3] - 139 y[4]
  z_1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   - 2 x[0] - x[1] - 10 x[2] + 8 x[3] + 116 y[0] + 126 y[1] - 144 y[2]
   - 32 y[3] - 75 y[4]
  z_2: Priority=0 Weight=1 AbsTol=0 RelTol=0
   7 x[0] + 4 x[1] + x[2] - 2 x[3] - 2 x[4] - 63 y[0] - 114 y[1] + 185 y[2]
   - 119 y[3] - 103 y[4]
Subject To
 con_0: 7 x[0] + 14 x[1] + 12 x[2] + 7 x[3] + 10 x[4] + 7 y[0] <= 68
 con_1: 17 x[0] + 10 x[1] + 7 x[2] + 6 x[3] + x[4] + 17 y[1] <= 70
 con_2: 16 x[0] + 10 x[1] + 20 x[2] + 14 x[3] + 19 x[4] + 12 y[2] <= 76
 con_3: 19 x[0] + 4 x[1] + 6 x[2] + 2 x[3] + 5 x[4] + 14 y[3] <= 73
 con_4: 3 x[0] + 9 x[1] + 10 x[2] + 18 x[3] + 6 x[4] + 8 y[4] <= 72
 con_5: 5 x[0] + 9 x[1] + 2 x[3] + 4 x[4] <= 93
 con_6: 3 x[0] + 13 x[1] + 5 x[2] + 18 x[3] + x[4] <= 87
 con_7: 15 x[0] + 19 x[1] + 13 x[2] + 11 x[3] + 17 x[4] <= 60
 con_8: 3 x[0] + 14 x[1] + 16 x[2] + 6 x[3] + 13 x[4] <= 58
 con_9: y[0] + y[1] + y[2] + y[3] + y[4] <= 1.666666666666667
Bounds
Binaries
 y[0] y[1] y[2] y[3] y[4]
End
