import numpy as np
sense_cross=np.load('hera127.drift_0_mod_0.150_cross.npz')
sense_auto=np.load('hera127.drift_mod_0.150_auto.npz')
print('1-cross_k/auto_k='+str(np.abs(1.-sense_cross['ks']/sense_auto['ks'])))
print('1-cross_errs/auto_errs='+str(np.abs(1.-sense_cross['errs']/sense_auto['errs'])))
