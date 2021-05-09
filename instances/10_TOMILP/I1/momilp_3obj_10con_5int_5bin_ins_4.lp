\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  z_0: Priority=2 Weight=1 AbsTol=0 RelTol=0
   89 y[0] - 198 y[1] + 146 y[2] - 24 y[3] - 109 y[4]
  z_1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   x[0] + 6 x[1] + 6 x[2] + x[3] - 5 x[4] + 108 y[0] - 44 y[1] + 32 y[2]
   + 115 y[3] + 77 y[4]
  z_2: Priority=0 Weight=1 AbsTol=0 RelTol=0
   - 6 x[0] + 6 x[1] + 8 x[2] - 10 x[3] + 6 x[4] + 42 y[0] - 164 y[1]
   - 28 y[2] - 18 y[3] - 87 y[4]
Subject To
 con_0: 9 x[0] + 2 x[1] + 7 x[2] - x[3] + 20 x[4] + 4 y[0] <= 90
 con_1: 18 x[0] + 9 x[1] + 10 x[2] + 8 x[3] + 9 x[4] + 13 y[1] <= 89
 con_2: 20 x[0] + 5 x[1] - x[2] + 19 x[3] + 11 x[4] <= 79
 con_3: 6 x[0] + 13 x[1] + 16 x[2] + x[3] + x[4] + 16 y[3] <= 82
 con_4: 18 x[1] + 4 x[2] + 7 x[3] + 13 x[4] <= 98
 con_5: 9 x[1] + 6 x[2] + 10 x[3] <= 59
 con_6: 14 x[0] + 15 x[1] + 4 x[2] + 19 x[3] + 16 x[4] <= 83
 con_7: 20 x[0] + 13 x[1] + 19 x[2] - x[3] - x[4] <= 97
 con_8: 8 x[0] + 17 x[1] + 19 x[2] + 4 x[3] + 6 x[4] <= 74
 con_9: y[0] + y[1] + y[2] + y[3] + y[4] <= 1.666666666666667
Bounds
Binaries
 y[0] y[1] y[2] y[3] y[4]
End
