\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  z_0: Priority=2 Weight=1 AbsTol=0 RelTol=0
   - 143 y[0] - 81 y[1] - 109 y[2] - 101 y[3] - 79 y[4]
  z_1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   - 6 x[0] - 2 x[1] + 5 x[2] - 3 x[4] + 91 y[0] + 67 y[1] + 184 y[2]
   - 147 y[3] - 116 y[4]
  z_2: Priority=0 Weight=1 AbsTol=0 RelTol=0
   - 5 x[0] + 10 x[1] - 6 x[2] - 9 x[3] - x[4] + 158 y[0] - 118 y[1]
   + 198 y[2] + 196 y[3] + 3 y[4]
Subject To
 con_0: 11 x[0] + 14 x[1] + 20 x[2] - x[3] + 2 x[4] + 2 y[0] <= 54
 con_1: 2 x[0] + 6 x[1] + 8 x[2] + 18 x[3] + 20 x[4] + 10 y[1] <= 56
 con_2: 17 x[0] + 3 x[1] + 5 x[2] + 11 x[3] + 17 y[2] <= 54
 con_3: 5 x[0] + 6 x[1] + 13 x[2] + 16 x[3] + 4 x[4] + y[3] <= 97
 con_4: 12 x[0] + 7 x[1] + 8 x[2] + 19 x[3] + 18 x[4] - y[4] <= 53
 con_5: 15 x[0] + 18 x[1] + 4 x[2] + 14 x[3] + 14 x[4] <= 62
 con_6: - x[0] + 17 x[1] + 2 x[2] + 16 x[3] + 18 x[4] <= 86
 con_7: 18 x[0] + 18 x[1] + 13 x[2] + 6 x[3] - x[4] <= 90
 con_8: 8 x[1] - x[2] + 9 x[3] + 19 x[4] <= 64
 con_9: y[0] + y[1] + y[2] + y[3] + y[4] <= 1.666666666666667
Bounds
Generals
 y[0] y[1] y[2] y[3] y[4]
End
