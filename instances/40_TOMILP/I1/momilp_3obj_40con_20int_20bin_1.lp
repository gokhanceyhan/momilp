\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  z_0: Priority=2 Weight=1 AbsTol=0 RelTol=0
   14 y[0] + 50 y[1] - 160 y[2] - 21 y[3] - 171 y[4] + 38 y[5] + 16 y[6]
   - 33 y[7] + 79 y[8] - 113 y[9] + 17 y[10] + 16 y[11] + 168 y[12]
   + 174 y[13] + 165 y[14] + 191 y[15] - 152 y[16] - 18 y[17] + 138 y[18]
   + 64 y[19]
  z_1: Priority=1 Weight=1 AbsTol=0 RelTol=0
   6 x[1] + 10 x[2] + 6 x[3] + 2 x[4] + 4 x[5] + 6 x[6] + 4 x[7] + 6 x[8]
   + 8 x[9] - 3 x[10] - 7 x[11] + 6 x[12] + 3 x[13] + x[14] + 7 x[15]
   - 9 x[16] - 4 x[17] - 6 x[18] + 116 y[0] - 107 y[1] - 12 y[2] + 96 y[3]
   + 25 y[4] - 35 y[5] + 72 y[6] - 43 y[7] + 93 y[9] + 145 y[10] - 99 y[11]
   - 157 y[12] + 39 y[13] + 195 y[14] - 81 y[15] - y[16] - 104 y[17]
   - 113 y[18] + 66 y[19]
  z_2: Priority=0 Weight=1 AbsTol=0 RelTol=0
   - 7 x[0] + 6 x[1] + 10 x[2] + 7 x[3] + 5 x[4] - 9 x[5] - 6 x[6] - 3 x[7]
   + 5 x[8] + x[9] - 8 x[10] - 3 x[11] - 5 x[12] - 5 x[13] - 2 x[14]
   - 5 x[15] - 5 x[16] - 2 x[17] + 10 x[18] - 7 x[19] + 29 y[0] - 31 y[1]
   + 9 y[2] - 141 y[3] - 106 y[4] + 182 y[5] + 155 y[6] - 135 y[7]
   + 184 y[8] + 167 y[9] - 66 y[10] + 97 y[11] - 90 y[12] + 78 y[13]
   - 189 y[14] + 82 y[15] + 39 y[16] + 156 y[17] - 198 y[18] + 61 y[19]
Subject To
 con_0: 11 x[0] + 14 x[1] + 20 x[2] - x[3] + 2 x[4] + 2 x[5] + 6 x[6]
   + 8 x[7] + 18 x[8] + 20 x[9] + 17 x[10] + 3 x[11] + 5 x[12] + 11 x[13]
   + 5 x[15] + 6 x[16] + 13 x[17] + 16 x[18] + 4 x[19] - y[0] <= 88
 con_1: 12 x[0] + 7 x[1] + 8 x[2] + 19 x[3] + 18 x[4] + 15 x[5] + 18 x[6]
   + 4 x[7] + 14 x[8] + 14 x[9] - x[10] + 17 x[11] + 2 x[12] + 16 x[13]
   + 18 x[14] + 18 x[15] + 18 x[16] + 13 x[17] + 6 x[18] - x[19] + 9 y[1]
   <= 88
 con_2: 8 x[1] - x[2] + 9 x[3] + 19 x[4] + 2 x[5] + 10 x[6] + 17 x[7]
   + x[8] - x[9] - x[10] + 3 x[11] + 20 x[12] + 4 x[13] + 5 x[14] + 7 x[15]
   + 19 x[16] + 16 x[17] + 14 x[18] + 3 x[19] + 11 y[2] <= 86
 con_3: 8 x[0] + 9 x[1] + 6 x[4] + 8 x[5] + 2 x[6] + 5 x[7] + 10 x[8]
   + 13 x[9] + 17 x[10] - x[11] + 13 x[12] + 2 x[13] + 20 x[14] + 11 x[15]
   + 9 x[16] + 19 x[17] + 10 x[18] + 3 x[19] - y[3] <= 51
 con_4: 5 x[0] + 3 x[1] + 14 x[2] + 19 x[3] + 2 x[4] + 11 x[5] + 3 x[6]
   + 19 x[7] + 7 x[8] + 13 x[9] + 14 x[10] + 19 x[11] + 2 x[12] + 14 x[13]
   + 12 x[14] + 20 x[15] + 20 x[16] + 15 x[17] + 16 x[18] + 4 x[19]
   + 6 y[4] <= 86
 con_5: 8 x[0] + 2 x[1] - x[2] + 4 x[3] - x[4] + 16 x[5] + 17 x[6] + 3 x[7]
   + x[8] + 15 x[9] + 2 x[10] + x[11] + 9 x[12] + 12 x[13] + 15 x[14]
   + 6 x[15] + 20 x[16] + 8 x[17] - x[18] + 9 x[19] + 18 y[5] <= 55
 con_6: 17 x[0] + 10 x[1] + x[2] + x[3] + 2 x[4] + 2 x[5] + 17 x[6]
   + 13 x[7] + 2 x[8] + 19 x[9] + 16 x[10] + 17 x[11] + 13 x[12] + 8 x[13]
   + 3 x[15] + 9 x[16] + 10 x[17] + 7 x[18] + 10 x[19] + 4 y[6] <= 55
 con_7: x[0] + 18 x[1] + 15 x[2] - x[3] - x[4] + 5 x[5] + 18 x[6] + 13 x[7]
   + 9 x[8] + 18 x[9] + 7 x[10] + 12 x[11] + x[12] + 2 x[13] + x[14]
   + 10 x[15] + 12 x[16] + 15 x[17] + 7 x[18] + 7 x[19] + 3 y[7] <= 88
 con_8: 18 x[0] + 7 x[1] + x[2] + 19 x[3] + 2 x[4] + 11 x[5] + 13 x[6]
   - x[7] + 3 x[8] + 2 x[9] + 12 x[10] + 10 x[11] + 12 x[12] + 12 x[13]
   + 10 x[14] + 15 x[15] + 20 x[16] + 13 x[17] + 15 x[18] + 20 x[19]
   + 18 y[8] <= 85
 con_9: 18 x[0] + 7 x[2] - x[3] + 3 x[4] + 5 x[5] + 20 x[6] + 12 x[7]
   + 6 x[8] + 14 x[9] + 8 x[10] + 17 x[11] + 7 x[12] + 14 x[13] + 10 x[14]
   + 5 x[15] + 14 x[16] + 11 x[18] + 2 x[19] + 3 y[9] <= 74
 con_10: 17 x[0] + 14 x[1] + 2 x[2] + 9 x[3] + 11 x[4] + 5 x[5] + 2 x[6]
   + 4 x[7] + 10 x[8] - x[9] + 10 x[10] + 7 x[11] + 19 x[12] + 9 x[13]
   + 10 x[14] + 4 x[15] + 14 x[16] + 7 x[17] + x[18] + 18 x[19] + 4 y[10]
   <= 99
 con_11: 20 x[0] + 18 x[1] + 13 x[2] + 19 x[3] + 20 x[4] + 2 x[5] + 2 x[6]
   + 6 x[7] + 8 x[8] + 8 x[9] + 8 x[10] + 2 x[11] + 13 x[12] + 17 x[13]
   + 11 x[14] + 2 x[15] + 8 x[16] + 9 x[17] + 12 x[18] + 6 x[19] + 11 y[11]
   <= 65
 con_12: 6 x[0] + 20 x[1] + 11 x[3] + x[4] + x[5] + 4 x[7] + 7 x[8]
   + 3 x[9] - x[10] + 10 x[11] + x[12] + 4 x[13] + 20 x[14] + 15 x[15]
   + 7 x[16] + 16 x[18] + 15 x[19] + 16 y[12] <= 70
 con_13: 2 x[0] + 7 x[1] + 19 x[2] + 3 x[3] + 15 x[4] + 2 x[5] + 6 x[6]
   + 2 x[7] + 17 x[8] + 16 x[9] + 17 x[11] + 16 x[12] + 19 x[13] + x[14]
   + 4 x[15] + 4 x[16] + 11 x[17] + 11 x[18] + 20 x[19] + 18 y[13] <= 100
 con_14: x[0] + 14 x[1] + 20 x[2] + 6 x[3] + 6 x[4] + 10 x[5] + 16 x[6]
   + 5 x[7] + 6 x[8] + 17 x[9] + 10 x[10] + 18 x[11] + 16 x[12] + 8 x[13]
   + 13 x[14] + 20 x[15] + 8 x[16] + 17 x[17] - x[18] + 8 x[19] + 11 y[14]
   <= 59
 con_15: 10 x[0] + 16 x[1] + 8 x[2] - x[3] + 15 x[4] + 9 x[5] + 3 x[6]
   + 19 x[7] + 2 x[8] + 18 x[9] + 7 x[10] + 13 x[11] - x[12] + 18 x[13]
   + 7 x[14] + 6 x[15] + 12 x[16] + 16 x[17] + 19 x[18] - x[19] + 5 y[15]
   <= 54
 con_16: 10 x[0] + 3 x[1] - x[2] + 9 x[3] + 13 x[4] + 19 x[5] + 17 x[6]
   + 3 x[7] + 2 x[8] + 18 x[9] + 6 x[10] + 7 x[11] + 12 x[12] + 4 x[13]
   - x[14] + 7 x[15] + 14 x[16] + 14 x[17] + 20 x[18] + 10 x[19] + 6 y[16]
   <= 99
 con_17: 3 x[0] + 6 x[1] + 12 x[2] + 9 x[3] + 19 x[4] + 13 x[5] + 16 x[6]
   + 18 x[7] + 2 x[8] + 8 x[9] + 17 x[10] + 20 x[11] + x[12] + 18 x[13]
   + 20 x[14] + 10 x[15] + 17 x[16] + 12 x[17] + 5 x[19] + 19 y[17] <= 74
 con_18: 9 x[0] + 20 x[1] + 15 x[2] + 15 x[3] + 2 x[4] + 16 x[5] + 9 x[6]
   + 8 x[7] + 11 x[8] + 12 x[9] + 5 x[10] + 5 x[11] + 6 x[12] + 7 x[13]
   + 12 x[14] + 6 x[15] + 15 x[16] + 14 x[17] + 7 x[18] + 2 x[19]
   + 15 y[18] <= 80
 con_19: 5 x[0] + 16 x[1] + 6 x[2] + 19 x[3] + x[4] + 15 x[5] + 17 x[6]
   + 12 x[7] + 3 x[8] + 3 x[9] + 11 x[10] + 16 x[11] + 19 x[12] + 4 x[14]
   + 18 x[16] + 20 x[17] + 18 x[18] + 6 x[19] - y[19] <= 52
 con_20: 8 x[0] + 16 x[1] + 8 x[2] - x[3] + 18 x[4] + 3 x[5] + 16 x[6]
   + 19 x[7] + 20 x[8] + 15 x[9] + 2 x[10] + 9 x[11] + 19 x[13] + 3 x[14]
   + 3 x[15] + 15 x[16] + 15 x[17] + 14 x[18] + 9 x[19] <= 69
 con_21: 12 x[0] + 7 x[1] + 10 x[2] + 19 x[3] + 5 x[4] + 8 x[5] + 14 x[6]
   + 2 x[7] + 18 x[8] + 11 x[9] + 17 x[10] + 16 x[11] + 9 x[12] + x[13]
   + 10 x[15] + 2 x[16] + 11 x[17] + 3 x[18] <= 53
 con_22: 16 x[0] - x[1] + 6 x[2] + 3 x[3] + 2 x[4] + 4 x[5] + 5 x[6]
   + 2 x[7] + x[8] + 7 x[9] + 3 x[11] - x[12] + 12 x[13] + 18 x[14]
   + 8 x[15] + 20 x[16] + 4 x[17] + 6 x[19] <= 63
 con_23: 7 x[0] + 19 x[1] + 6 x[2] + 2 x[3] + 20 x[4] + 11 x[5] + 2 x[6]
   + 5 x[7] + 19 x[8] + 6 x[9] + 18 x[10] - x[11] + 4 x[12] + 10 x[13]
   + 8 x[14] + 18 x[15] + 6 x[16] + 20 x[17] + 4 x[18] + 7 x[19] <= 61
 con_24: 10 x[0] + 13 x[1] - x[2] + 7 x[3] + 18 x[4] + 5 x[5] + 9 x[6]
   + 18 x[7] + x[8] - x[9] + 18 x[10] + 13 x[11] - x[12] + 18 x[13]
   + 5 x[14] + 11 x[16] + 17 x[17] + 12 x[18] + 8 x[19] <= 63
 con_25: 3 x[0] + 16 x[1] + 2 x[2] + x[3] + 19 x[4] + 12 x[5] + 14 x[6]
   + 3 x[7] + 8 x[8] + 3 x[9] + 16 x[10] + 17 x[11] + 12 x[12] + 12 x[13]
   + x[14] + 2 x[15] + 8 x[16] + 11 x[17] + 14 x[18] + x[19] <= 86
 con_26: 18 x[0] + 5 x[1] + 13 x[2] + 15 x[3] + 14 x[4] + 7 x[5] - x[6]
   + 9 x[7] + 10 x[8] + 12 x[9] + 6 x[10] + 5 x[11] + 12 x[12] + 20 x[13]
   + 14 x[14] + 8 x[15] + 5 x[16] + 20 x[17] + 10 x[18] + 17 x[19] <= 67
 con_27: 6 x[0] + 8 x[2] + x[3] + 17 x[4] + 13 x[5] + 4 x[6] + 3 x[7]
   + 17 x[8] + 13 x[9] + 17 x[10] + 11 x[12] - x[13] + 8 x[14] + 15 x[15]
   + x[16] + 7 x[17] + 18 x[18] + 9 x[19] <= 76
 con_28: 15 x[0] - x[2] + 4 x[3] + x[4] + 14 x[5] + 2 x[6] + 13 x[7]
   + 4 x[8] + 10 x[9] + 18 x[10] + 5 x[11] + 19 x[12] + 5 x[13] + 18 x[14]
   + 5 x[15] + x[16] + 5 x[17] + 4 x[18] + 4 x[19] <= 56
 con_29: 12 x[0] + 8 x[1] + 3 x[2] + 12 x[3] + 20 x[4] + 16 x[5] + 2 x[6]
   + 12 x[7] + 18 x[8] + 7 x[9] + 20 x[10] + 12 x[11] + 4 x[12] + 15 x[13]
   + 13 x[14] + 4 x[15] + 4 x[17] + 5 x[18] + 11 x[19] <= 97
 con_30: 9 x[0] + 13 x[1] + 7 x[2] + 6 x[3] + 12 x[4] + 4 x[5] + 14 x[6]
   + 9 x[7] + 2 x[8] + 9 x[9] + 17 x[10] + 2 x[11] + 13 x[12] + 17 x[13]
   + 4 x[14] + 3 x[15] + 16 x[16] + 4 x[17] + 2 x[18] + 4 x[19] <= 84
 con_31: 7 x[0] + 3 x[1] + 16 x[2] + 17 x[4] + 13 x[6] + 16 x[7] + 6 x[8]
   + 20 x[9] + 10 x[10] + 13 x[11] + 15 x[12] + 19 x[13] + 16 x[15]
   + 11 x[16] + 13 x[17] - x[18] + x[19] <= 71
 con_32: 18 x[0] + 13 x[1] + 17 x[2] + 10 x[3] + 19 x[4] + 8 x[5] + 13 x[6]
   - x[7] + 11 x[8] + 8 x[9] + 17 x[10] + 3 x[11] + 6 x[12] + 2 x[13]
   + 15 x[14] + 11 x[15] + 4 x[16] + 19 x[17] + 12 x[18] + 15 x[19] <= 70
 con_33: 12 x[0] + 17 x[1] + 18 x[2] + 16 x[3] + 6 x[4] + 16 x[5] + 13 x[6]
   + 2 x[7] + 9 x[8] + 19 x[9] + 9 x[10] + 19 x[12] + 15 x[13] + 9 x[14]
   + 17 x[15] + 19 x[16] + 15 x[17] + x[18] + 5 x[19] <= 92
 con_34: 8 x[0] + x[1] + 19 x[2] + 11 x[3] + 8 x[4] + 20 x[5] + 3 x[6]
   + 3 x[7] + 8 x[8] + 9 x[9] + 11 x[10] + 7 x[11] + 4 x[13] + 6 x[14]
   - x[15] + 2 x[17] + 17 x[18] + 17 x[19] <= 67
 con_35: 19 x[0] + 7 x[1] + 17 x[2] + 6 x[3] + 9 x[4] + 17 x[5] + 2 x[6]
   + 11 x[7] + 18 x[8] + 12 x[9] - x[10] + 9 x[11] + 2 x[12] + 5 x[13]
   + 2 x[14] + 5 x[15] + 18 x[16] + x[17] + 17 x[18] + 17 x[19] <= 87
 con_36: 5 x[0] + 19 x[1] + 10 x[2] - x[3] + 3 x[4] + 12 x[5] + 8 x[6]
   + x[7] + 11 x[9] + 14 x[10] + 11 x[12] + 4 x[13] + 5 x[14] + 9 x[16]
   + 5 x[17] + 5 x[18] + 5 x[19] <= 75
 con_37: 5 x[0] + 10 x[1] + 17 x[2] + 20 x[3] + 17 x[4] + 17 x[5] + 18 x[6]
   + x[7] + 11 x[8] + 8 x[9] + 18 x[10] + 7 x[11] + 4 x[12] + 19 x[13]
   + 20 x[14] + 10 x[15] + 10 x[17] + 14 x[18] + 11 x[19] <= 71
 con_38: 14 x[0] + 11 x[1] + 20 x[2] + 17 x[3] + 12 x[4] + 11 x[5] + 8 x[6]
   + 20 x[7] + 5 x[8] + 3 x[9] + 15 x[10] + 7 x[11] + 4 x[12] + 14 x[13]
   + 4 x[14] + 4 x[15] + 16 x[16] - x[17] + 15 x[18] + 15 x[19] <= 86
 con_39: y[0] + y[1] + y[2] + y[3] + y[4] + y[5] + y[6] + y[7] + y[8]
   + y[9] + y[10] + y[11] + y[12] + y[13] + y[14] + y[15] + y[16] + y[17]
   + y[18] + y[19] <= 6.666666666666667
Bounds
Binaries
 y[0] y[1] y[2] y[3] y[4] y[5] y[6] y[7] y[8] y[9] y[10] y[11] y[12] y[13]
 y[14] y[15] y[16] y[17] y[18] y[19]
End
