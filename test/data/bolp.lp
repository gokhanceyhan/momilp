\ Model multiobj
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize multi-objectives
  Set1: Priority=2 Weight=1 AbsTol=0 RelTol=0.01
   5 El[5] + 6 El[6] + 7 El[7] + 8 El[8] + 9 El[9] + 15 El[15] + 16 El[16] + 17 El[17] + 18 El[18]
   + 19 El[19]
  Set2: Priority=1 Weight=1 AbsTol=0 RelTol=0.01
   33 El[3] + 44 El[4] + 5 El[5] + 9 El[9] + El[10] + El[11] + El[15] + El[16] + El[17]
Subject To
 Budget: El[0] + El[1] + El[2] + El[3] + El[4] + El[5] + El[6] + El[7]
   + El[8] + El[9] + El[10] + El[11] + El[12] + El[13] + El[14] + El[15]
   + El[16] + El[17] + El[18] + El[19] <= 10
Bounds
El[0] <= 1
El[1] <= 1
El[2] <= 1
El[3] <= 1
El[4] <= 1
El[5] <= 1
El[6] <= 1
El[7] <= 1
El[8] <= 1
El[9] <= 1
El[10] <= 1
El[11] <= 1
El[12] <= 1
El[13] <= 1
El[14] <= 1
El[15] <= 1
El[16] <= 1
El[17] <= 1
El[18] <= 1
El[19] <= 1
End
