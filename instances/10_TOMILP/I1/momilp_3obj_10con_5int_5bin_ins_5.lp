\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  z_0: Priority=2 Weight=1 AbsTol=0 RelTol=0
   - 149 y[0] + 128 y[1] - 146 y[2] + 188 y[3] + 64 y[4]
  z_1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   x[0] - 3 x[1] - 3 x[2] + 9 x[3] - 6 x[4] + 64 y[0] + 50 y[1] + 152 y[2]
   - 146 y[3] - 169 y[4]
  z_2: Priority=0 Weight=1 AbsTol=0 RelTol=0
   8 x[0] + 3 x[1] - x[2] + 6 x[3] + x[4] - 35 y[0] + 181 y[1] - 42 y[2]
   - 125 y[3] + 96 y[4]
Subject To
 con_0: 13 x[0] + 4 x[1] + 7 x[3] + 7 x[4] + 16 y[0] <= 67
 con_1: 17 x[0] + 8 x[1] + 6 x[2] + 12 x[3] + 7 x[4] + 8 y[1] <= 64
 con_2: 3 x[0] + 17 x[1] + 11 x[2] + 5 x[3] + 9 x[4] + 15 y[2] <= 59
 con_3: 19 x[0] + 2 x[1] - x[2] + 20 x[3] + 20 x[4] + 11 y[3] <= 54
 con_4: 8 x[0] + 5 x[1] + 5 x[2] + x[3] + 13 x[4] + 17 y[4] <= 100
 con_5: 20 x[0] + 7 x[1] + 16 x[2] + x[3] - x[4] <= 53
 con_6: 7 x[0] + 18 x[1] + 9 x[2] + 12 x[3] <= 54
 con_7: - x[0] + 12 x[1] + 2 x[2] + 17 x[3] + 18 x[4] <= 89
 con_8: 14 x[0] + x[1] + 17 x[2] + 16 x[3] + 15 x[4] <= 56
 con_9: y[0] + y[1] + y[2] + y[3] + y[4] <= 1.666666666666667
Bounds
Binaries
 y[0] y[1] y[2] y[3] y[4]
End
