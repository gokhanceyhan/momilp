\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  z_0: Priority=2 Weight=1 AbsTol=0 RelTol=0
  
  z_1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   - 10 x[0] - 4 x[1] + 4 x[2] + 9 x[3] - 10 x[4] - 8 x[5] + x[6] + 4 x[7]
   - 5 x[8] - 7 x[9] - 4 x[10] - 8 x[11] - 6 x[12] + 7 x[13] + 10 x[14]
   - x[15] + 5 x[16] + 2 x[17] + 10 x[18] + 9 x[19] + 56 y[0] + 181 y[1]
   + 184 y[2] + 114 y[3] - 25 y[4] + 9 y[5] + 55 y[6] + 128 y[7] - 120 y[8]
   + 94 y[9] + 69 y[10] - 156 y[11] - 14 y[12] + 165 y[13] - 117 y[14]
   - 154 y[15] - 140 y[16] + 124 y[17] - 17 y[18] + 149 y[19]
  z_2: Priority=0 Weight=1 AbsTol=0 RelTol=0
   - x[0] + 6 x[1] + 6 x[2] - x[3] + 6 x[4] - 4 x[5] + x[6] + 3 x[7] + 4 x[8]
   + 6 x[9] - 10 x[10] + 4 x[11] - 8 x[12] + 5 x[13] + 5 x[14] - 6 x[15]
   + 8 x[16] - 5 x[17] + 9 x[18] - 6 x[19] + 20 y[0] - 42 y[1] + 112 y[2]
   - 173 y[3] + 31 y[4] - 13 y[5] + 174 y[6] - 168 y[7] + 109 y[8]
   - 86 y[9] - 25 y[10] - 35 y[11] - 179 y[12] - 115 y[13] - 122 y[14]
   - 76 y[15] + 95 y[16] + 61 y[17] - 148 y[18] + 176 y[19]
Subject To
 con_0: 6 x[0] + 15 x[3] + 19 x[5] + 7 x[7] + 11 x[9] + 2 x[10] + 7 x[11]
   + 17 x[14] + 11 x[15] + 16 x[16] + 10 x[17] + 12 x[18] + 15 y[0] <= 56
 con_1: 14 x[0] + 9 x[7] + 6 x[9] + 14 x[10] + 16 x[11] + 11 x[13]
   + 12 x[14] + 16 x[17] + 3 x[18] + 6 x[19] + 5 y[1] <= 100
 con_2: 11 x[0] + 16 x[1] + 13 x[3] + 6 x[5] + 12 x[10] + 11 x[14]
   + 4 x[16] + 13 x[18] + 12 x[19] + 5 y[2] <= 62
 con_3: 16 x[2] + 2 x[4] + 14 x[6] + 3 x[13] + x[17] + 13 y[3] <= 95
 con_4: - x[1] + 9 x[3] + 20 x[6] + 10 x[7] + 12 x[8] + 13 x[11] + 6 x[12]
   + x[13] + 14 x[16] + 6 x[17] + 13 x[19] + 20 y[4] <= 58
 con_5: 11 x[5] + 8 x[6] + 6 x[10] + 12 x[12] + 7 x[13] + 10 x[14]
   + 18 y[5] <= 54
 con_6: 13 x[3] + 13 x[5] + 10 x[6] + 8 x[10] + 18 x[14] + 16 x[18]
   + 19 y[6] <= 100
 con_7: 19 x[0] + 19 x[1] + 20 x[4] + 3 x[5] + 9 x[7] + 8 x[8] + 20 x[9]
   + 10 x[10] + x[11] + 19 x[12] + 11 x[13] + 16 x[17] + 8 y[7] <= 79
 con_8: x[1] + 15 x[4] + 8 x[6] + 7 x[7] + 10 x[8] + 14 x[10] + 9 x[18]
   + 18 x[19] + 7 y[8] <= 57
 con_9: 17 x[3] + 8 x[10] + 16 x[12] + x[15] + 15 x[16] + 8 x[17]
   + 13 x[19] + 19 y[9] <= 64
 con_10: 11 x[4] + 12 x[8] + 10 x[9] + 19 x[10] + 9 x[13] + 9 x[15]
   + 9 x[16] + 17 x[17] + 11 x[19] + 10 y[10] <= 83
 con_11: 20 x[0] + x[1] + 10 x[6] + 16 x[7] + 15 x[8] + 7 x[12] + 12 x[13]
   + 16 x[15] + 20 x[16] + 5 x[17] + 4 x[19] + 2 y[11] <= 87
 con_12: 12 x[8] - x[14] - y[12] <= 61
 con_13: - x[1] + 16 x[3] + 2 x[6] + 15 x[7] - x[8] + 13 x[11] + 20 x[14]
   + 5 x[16] + 6 x[17] + 2 y[13] <= 63
 con_14: x[5] + 16 x[6] + 3 x[9] + 4 x[13] + 14 x[17] + 18 x[19] + 3 y[14]
   <= 91
 con_15: 10 x[1] + 16 x[3] + 17 x[4] + 14 x[5] + 16 x[12] + 16 x[13]
   + 20 x[16] + 18 y[15] <= 69
 con_16: 13 x[5] + 15 x[6] + 8 x[7] - x[13] + 7 x[14] + 12 x[15] + 10 y[16]
   <= 74
 con_17: 11 x[1] + 11 x[2] + 6 x[8] + 19 x[10] + 7 x[11] + 14 x[12]
   + 20 x[14] + 16 x[16] + 18 y[17] <= 78
 con_18: 15 x[0] + 11 x[1] + 12 x[4] + 4 x[12] + x[14] + 18 x[15] + x[18]
   + 8 x[19] + 5 y[18] <= 53
 con_19: 15 x[1] + 12 x[3] + 11 x[6] + 12 x[8] + 14 x[13] + 4 x[15]
   + 3 y[19] <= 53
 con_20: 20 x[1] + 10 x[3] + x[5] + 18 x[6] + 5 x[7] + x[9] - x[10]
   + 18 x[11] + 10 x[14] + 4 x[15] + 9 x[17] <= 60
 con_21: 10 x[2] + 8 x[8] - x[9] + 14 x[10] + 11 x[11] + 8 x[12] + 16 x[14]
   + 19 x[15] <= 68
 con_22: x[0] + 8 x[1] + 20 x[2] + 19 x[5] + 19 x[7] + 20 x[8] + 15 x[9]
   + 4 x[16] + 11 x[17] <= 92
 con_23: 10 x[0] + 19 x[4] + 4 x[5] + 13 x[9] + 17 x[11] + 10 x[15]
   + 9 x[17] <= 59
 con_24: 15 x[0] + 9 x[1] + 9 x[5] + 2 x[8] - x[14] + 16 x[15] + 12 x[16]
   + 9 x[17] + 14 x[18] <= 74
 con_25: 15 x[0] + 8 x[1] - x[4] + 11 x[5] + 18 x[7] + 10 x[8] + 19 x[10]
   + 16 x[11] + 7 x[13] + 9 x[15] + 12 x[16] + 17 x[17] + 17 x[18]
   + 10 x[19] <= 80
 con_26: 11 x[0] + x[1] + 4 x[3] + 8 x[8] + 15 x[9] + 7 x[11] + 20 x[13]
   + 17 x[14] + 3 x[15] + 9 x[16] <= 50
 con_27: 4 x[1] + 20 x[3] + 14 x[5] + x[6] + 8 x[9] + 19 x[10] + 12 x[11]
   + 3 x[15] + 10 x[16] + 4 x[19] <= 61
 con_28: 2 x[3] + 4 x[5] + 14 x[8] + x[11] + 14 x[12] + 17 x[13] + 11 x[16]
   + 18 x[17] - x[18] + 13 x[19] <= 67
 con_29: 13 x[1] + 20 x[3] + 19 x[5] + 8 x[6] + 20 x[7] + 8 x[17]
   + 20 x[18] <= 68
 con_30: 11 x[0] + 6 x[1] + 9 x[2] + 9 x[3] + 5 x[4] + 7 x[5] + 16 x[7]
   + 8 x[8] + 3 x[9] + 11 x[10] + 6 x[12] + 7 x[14] - x[15] + 14 x[16]
   + 3 x[17] + 11 x[19] <= 79
 con_31: 14 x[1] + 19 x[8] + 5 x[9] + 17 x[11] + 17 x[12] + 13 x[15]
   + 16 x[19] <= 98
 con_32: x[3] + 15 x[4] + 9 x[6] + 15 x[7] + 14 x[9] + 16 x[14] <= 67
 con_33: 9 x[1] + 6 x[3] + 14 x[4] - x[7] + 20 x[8] + 4 x[10] + 8 x[11]
   + 16 x[15] + 9 x[16] + x[17] + 19 x[18] + 5 x[19] <= 65
 con_34: 6 x[0] + 12 x[2] + 14 x[7] + 10 x[8] + 17 x[9] + 11 x[12]
   + 4 x[13] + 15 x[14] + 3 x[15] + 8 x[16] + 6 x[18] <= 92
 con_35: 8 x[6] + 5 x[7] + 10 x[8] + 6 x[9] + 3 x[12] + 6 x[14] + 19 x[15]
   + 10 x[19] <= 50
 con_36: 11 x[1] - x[6] + x[7] + 9 x[8] + x[9] + 6 x[10] + 3 x[11]
   + 16 x[16] + 19 x[19] <= 95
 con_37: 5 x[1] + 11 x[2] + 10 x[5] + 20 x[6] + 16 x[8] + 10 x[9] + 9 x[13]
   + 9 x[14] + 19 x[15] + x[18] <= 65
 con_38: 15 x[7] + 13 x[9] + 14 x[11] + 15 x[16] + 3 x[17] + 2 x[18] <= 79
 con_39: y[0] + y[1] + y[2] + y[3] + y[4] + y[5] + y[6] + y[7] + y[8]
   + y[9] + y[10] + y[11] + y[12] + y[13] + y[14] + y[15] + y[16] + y[17]
   + y[18] + y[19] <= 6.666666666666666
Bounds
Binaries
 y[0] y[1] y[2] y[3] y[4] y[5] y[6] y[7] y[8] y[9] y[10] y[11] y[12] y[13]
 y[14] y[15] y[16] y[17] y[18] y[19]
End
