#!/bin/bash
source activate analysis
python calc_sense.py -b 0.1 --eor=ps21_auto_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2 hera127.drift_blmin0_blmax84_0.150GHz_arrayfile.npz hera127.drift_blmin0_blmax84_0.150GHz_arrayfile.npz
python calc_sense_auto.py -b 0.1 --eor=ps21_auto_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2 hera127.drift_blmin0_blmax84_0.150GHz_arrayfile.npz 
