\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  z_0: Priority=2 Weight=1 AbsTol=0 RelTol=0
  
  z_1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   - 7 x[0] - 8 x[1] + 10 x[2] - 4 x[3] + 3 x[4] - 6 x[5] - x[6] + 5 x[7]
   + 7 x[8] - 5 x[9] + 23 y[0] - 61 y[1] - 71 y[2] - 157 y[3] + 183 y[4]
   - 37 y[5] + 155 y[6] - 103 y[7] + 83 y[8] + 103 y[9]
  z_2: Priority=0 Weight=1 AbsTol=0 RelTol=0
   9 x[0] - 9 x[1] - 2 x[2] - 3 x[3] + 5 x[4] - 3 x[5] + 3 x[6] - 3 x[7]
   + 3 x[8] - 5 x[9] - 192 y[0] + 97 y[1] + 194 y[2] + 195 y[3] + 43 y[4]
   + 20 y[5] + 75 y[6] + 161 y[7] - 147 y[8] - 59 y[9]
Subject To
 con_0: 7 x[0] + 20 x[1] + 16 x[2] + 20 x[6] + 12 x[7] + 2 x[9] + 11 y[0]
   <= 67
 con_1: 8 x[0] + 18 x[5] + 18 x[7] + 8 x[8] + 16 y[1] <= 67
 con_2: 14 x[0] + 2 x[7] + 16 x[8] + 8 y[2] <= 53
 con_3: 9 x[6] + 2 y[3] <= 62
 con_4: 13 x[0] + 7 x[1] + 18 x[2] + 10 x[4] + 18 x[7] + 19 x[8] + 18 y[4]
   <= 89
 con_5: 6 x[0] + 17 x[1] + 20 x[6] + 3 x[8] + 10 y[5] <= 85
 con_6: 8 x[1] + 18 x[8] + 7 x[9] + 20 y[6] <= 67
 con_7: 16 x[1] + 6 x[3] + 5 x[6] + 18 x[8] + 19 x[9] + 18 y[7] <= 90
 con_8: 12 x[3] - x[5] + 6 y[8] <= 88
 con_9: 7 x[2] + 10 x[6] + 9 y[9] <= 81
 con_10: - x[0] + 15 x[2] + 2 x[3] + 9 x[5] + 20 x[6] + 13 x[8] + 13 x[9]
   <= 55
 con_11: 6 x[1] - x[2] + x[3] + 6 x[5] + 15 x[8] <= 84
 con_12: 2 x[0] + 19 x[4] + 19 x[5] + 12 x[7] + 5 x[8] + 10 x[9] <= 96
 con_13: 5 x[0] + 9 x[3] + 4 x[8] <= 72
 con_14: 12 x[3] + x[4] + 11 x[5] + 19 x[9] <= 68
 con_15: 9 x[0] + 11 x[3] + 11 x[5] + 10 x[9] <= 64
 con_16: 9 x[1] + 19 x[2] + 12 x[4] + 2 x[8] + 11 x[9] <= 88
 con_17: 7 x[0] + 11 x[2] + 13 x[8] <= 52
 con_18: 10 x[1] + 19 x[4] + 13 x[9] <= 70
 con_19: y[0] + y[1] + y[2] + y[3] + y[4] + y[5] + y[6] + y[7] + y[8]
   + y[9] <= 3.333333333333333
Bounds
Binaries
 y[0] y[1] y[2] y[3] y[4] y[5] y[6] y[7] y[8] y[9]
End
