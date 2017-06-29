#! /usr/bin/env python
'''
Calculates the expected sensitivity of a 21cm experiment to a given 21cm power spectrum.  Requires as input an array .npz file created with mk_array_file.py.
'''
import aipy as a, numpy as n, optparse, sys
from scipy import interpolate

o = optparse.OptionParser()
o.set_usage('calc_sense.py [options] <array1>.npz <array2>.npz')
o.set_description(__doc__)
o.add_option('-m', '--model', dest='model', default='mod',
    help="The model of the foreground wedge to use.  Three options are 'pess' (all k modes inside horizon + buffer are excluded, and all baselines are added incoherently), 'mod' (all k modes inside horizon + buffer are excluded, but all baselines within a uv pixel are added coherently), and 'opt' (all modes k modes inside the primary field of view are excluded).  See Pober et al. 2014 for more details.")
o.add_option('-b', '--buff', dest='buff', default=0.1, type=float,
    help="The size of the additive buffer outside the horizon to exclude in the pessimistic and moderate models.")
o.add_option('--eor', dest='eor', default='ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2',
    help="The model epoch of reionization power spectrum.  The code is built to handle output power spectra from 21cmFAST.")
o.add_option('--ndays', dest='ndays', default=180., type=float,
    help="The total number of days observed.  The default is 180, which is the maximum a particular R.A. can be observed in one year if one only observes at night.  The total observing time is ndays*n_per_day.")
o.add_option('--n_per_day', dest='n_per_day', default=6., type=float,
    help="The number of good observing hours per day.  This corresponds to the size of a low-foreground region in right ascension for a drift scanning instrument.  The total observing time is ndays*n_per_day.  Default is 6.  If simulating a tracked scan, n_per_day should be a multiple of the length of the track (i.e. for two three-hour tracks per day, n_per_day should be 6).")
o.add_option('--bwidth', dest='bwidth', default=0.008, type=float,
             help="Cosmological bandwidth in GHz for a 21cm observation.  Note this is not the total instrument bandwidth, but the redshift range that can be considered co-eval.  Default is 0.008 (8 MHz).")
o.add_option('--nchan', dest='nchan', default=82, type=int,
    help="Integer number of channels across cosmological bandwidth.  Defaults to 82, which is equivalent to 1024 channels over 100 MHz of bandwidth.  Sets maximum k_parallel that can be probed, but little to no overall effect on sensitivity.")
o.add_option('--no_ns', dest='no_ns', action='store_true',
    help="Remove pure north/south baselines (u=0) from the sensitivity calculation.  These baselines can potentially have higher systematics, so excluding them represents a conservative choice.")
opts, args = o.parse_args(sys.argv[1:])

#=========================COSMOLOGY/BINNING FUNCTIONS=========================
F21 = 1.42040575177
FCO10 = 115.271208
#Convert frequency (GHz) to redshift for 21cm line.
def f2z(fq,line_freq=F21):
    return (line_freq / fq - 1)

#Multiply by this to convert an angle on the sky to a transverse distance in Mpc/h at redshift z
def dL_dth(z):
    '''[h^-1 Mpc]/radian, from Furlanetto et al. (2006)'''
    return 1.9 * (1./a.const.arcmin) * ((1+z) / 10.)**.2

#Multiply by this to convert a bandwidth in GHz to a line of sight distance in Mpc/h at redshift z
def dL_df(z, omega_m=0.266,line_freq=F21):
    '''[h^-1 Mpc]/GHz, from Furlanetto et al. (2006)'''
    return (1.7 / 0.1) * ((1+z) / 10.)**.5 * (omega_m/0.15)**-0.5 * 1e3 * F21/line_freq

#Multiply by this to convert a baseline length in wavelengths (at the frequency corresponding to redshift z) into a tranverse k mode in h/Mpc at redshift z
def dk_du(z):
    '''2pi * [h Mpc^-1] / [wavelengths], valid for u >> 1.'''
    return 2*n.pi / dL_dth(z) # from du = 1/dth, which derives from du = d(sin(th)) using the small-angle approx

#Multiply by this to convert eta (FT of freq.; in 1/GHz) to line of sight k mode in h/Mpc at redshift z
def dk_deta(z,line_freq=F21):
    '''2pi * [h Mpc^-1] / [GHz^-1]'''
    return 2*n.pi / dL_df(z,line_freq)

#scalar conversion between observing and cosmological coordinates
def X2Y(z,line_freq=F21):
    '''[h^-3 Mpc^3] / [str * GHz]'''
    return dL_dth(z)**2 * dL_df(z,line_freq)

#A function used for binning
def find_nearest(array,value):
    idx = (n.abs(array-value)).argmin()
    return idx



nchan = opts.nchan
#====================OBSERVATION/COSMOLOGY PARAMETER VALUES====================

#Load in data from array file; see mk_array_file.py for definitions of the parameters
array_dict={}
for argnum,arg in enumerate(args):
    array = n.load(args[0])
    name = str(array['name'])
    obs_duration = array['obs_duration']
    array_dict[name]={}
    array_dict[name]['obs_duration']=obs_duration
    dish_size_in_lambda = array['dish_size_in_lambda']
    array_dict[name]['dish_size_in_lambda']=dish_size_in_lambda
    Trx = array['Trx']
    array_dict[name]['Trx']=Trx
    t_int = array['t_int']
    array_dict[name]['t_int']=t_int
    if opts.model == 'pess':
        uv_coverage = array['uv_coverage_pess']
    else:
        uv_coverage = array['uv_coverage']
    uv_coverage *= t_int
    SIZE = uv_coverage.shape[0]
    # Cut unnecessary data out of uv coverage: auto-correlations & half of uv plane (which is not statistically independent for real sky)
    uv_coverage[SIZE/2,SIZE/2] = 0.
    uv_coverage[:,:SIZE/2] = 0.
    uv_coverage[SIZE/2:,SIZE/2] = 0.    
    if opts.no_ns: uv_coverage[:,SIZE/2] = 0.
    nonzero=n.where(uv_coverage>0)
    array_dict[name]['nonzero']=nonzero
    array_dict[name]['SIZE']=SIZE
    array_dict[name]['uv_coverage']=uv_coverage  
    obstype=str(array['linetype'])
    h = 0.7
    line_freq={'21cm':F21,'co':FCO10}[obstype]
    array_dict[name]['line_freq']=line_freq
    B = opts.bwidth*line_freq/F21
    array_dict[name]['B']=B
    z = f2z(array['freq'],line_freq)
    array_dict[name]['z']=z
    dish_size_in_lambda = dish_size_in_lambda*(array['freq']/.150) # linear frequency evolution, relative to 150 MHz
    array_dict[name]['dish_size_in_lambda']=dish_size_in_lambda
    first_null = 1.22/dish_size_in_lambda #for an airy disk, even though beam model is Gaussian
    array_dict[name]['first_null']=first_null
    bm = 1.13*(2.35*(0.45/dish_size_in_lambda))**2
    array_dict[name]['bm']=bm
    kpls = dk_deta(z,line_freq) * n.fft.fftfreq(nchan,B/nchan)
    array_dict[name]['kpls']=kpls
    Tsky = 60e3 * (3e8/(array['freq']*1e9))**2.55+2.7e3  # sky temperature in mK (added CMB for high frequencies)
    array_dict[name]['Tsky']=Tsky
    n_lstbins = opts.n_per_day*60./obs_duration
    array_dict[name]['n_lstbins']=n_lstbins

min_dish_size_in_lambda=9e99
for name in array_dict.keys():
    if array_dict[name]['dish_size_in_lambda']<min_dish_size_in_lambda:
        min_name=name
        min_dish_size_in_lambda=array_dict[name]['dish_size_in_lambda']

if array_dict.keys().index(min_name)==0:
    max_name=array_dict.keys()[1]
    max_num=1
    min_num=0
else:
    max_name=array_dict.keys()[0]
    max_num=0
    min_num=1

#===============================EOR MODEL===================================

#You can change this to have any model you want, as long as mk, mpk and p21 are returned

#This is a dimensionless power spectrum, i.e., Delta^2
modelfile = opts.eor
model = n.loadtxt(modelfile)
mk, mpk1, mpk2, mpk12 = model[:,0]/h, model[:,1], model[:,2], model[:,3] #k, Delta^2(k)
#note that we're converting from Mpc to h/Mpc

#interpolation function for the EoR model
p1 = interpolate.interp1d(mk, mpk1, kind='linear')
p2 = interpolate.interp1d(mk, mpk2, kind='linear')
p12 = interpolate.interp1d(mk, mpk12, kind='linear')

#=================================MAIN CODE===================================

#set up blank arrays/dictionaries
kprs = []
#sense will include sample variance, Tsense will be Thermal only
sense, Tsense, corrSense = {}, {}, {}
    
u_larger=array_dict[max_name]['nonzero'][1]-array_dict[max_name]['SIZE']/2
v_larger=array_dict[max_name]['nonzero'][0]-array_dict[max_name]['SIZE']/2
u_larger*=array_dict[max_name]['dish_size_in_lambda']
v_larger*=array_dict[max_name]['dish_size_in_lambda']

n_corr=np.round((array_dict[max_name]['dish_size_in_lambda']/array_dict[min_name]['dish_size_in_lambda'])**2.)
lst_ratio=np.round(array_dict[max_name]['dish_size_in_lambda']/array_dict[min_name]['dish_size_in_lambda'])

#loop over uv_coverage to calculate k_pr
for iu,iv in zip(array_dict[min_name]['nonzero'][1], array_dict[min_name]['nonzero'][0]):
   u, v = (iu - array_dict[min_name]['SIZE']/2) * array_dict[min_name]['dish_size_in_lambda'], (iv - array_dict['SIZE']/2) * array_dict[min_name]['dish_size_in_lambda']
   #find matching measurement in grid with larger uv cells.
   u_match=np.abs(u_larger-u)<array_dict[max_name]['dish_size_in_lambda']
   v_match=np.abs(v_larger-v)<array_dict[max_name]['dish_size_in_lambda']
   if len(u_match[u_match])==1 and len(v_match[v_match])==1:
       iu1=array_dict[max_name]['nonzero'][1][u_match]
       iv1=array-dict[max_name]['nonzero'][0][v_match]
       umag = n.sqrt(u**2 + v**2)
       kpr = umag * dk_du(z)
       kprs.append(kpr)
       #calculate horizon limit for baseline of length umag
       if opts.model in ['mod','pess']: hor = dk_deta(z) * umag/array['freq'] + opts.buff
       elif opts.model in ['opt']: hor = dk_deta(z) * (umag/array['freq'])*n.sin(first_null/2)
       else: print '%s is not a valid foreground model; Aborting...' % opts.model; sys.exit()
       if not sense.has_key(kpr): 
           sense[kpr] = n.zeros_like(kpls)
           Tsense[kpr] = n.zeros_like(kpls)
           corrSense[kpr]=n.zeros_like(kpls)
       for i, kpl in enumerate(kpls):
           #exclude k_parallel modes contaminated by foregrounds
           if n.abs(kpl) < hor: continue
           k = n.sqrt(kpl**2 + kpr**2)
           if k < min(mk): continue
           #don't include values beyond the interpolation range (no sensitivity anyway)
           if k > n.max(mk): continue
           tot_integration_1 = array_dict[min_name]['uv_coverage'][iv,iu] * opts.ndays
           tot_integration_2 = array_dict[max_name]['uv_coverage'][iv1,iu1] * opts.ndays
           delta1 = p1(k)
           delta2 = p2(k)
           delta12 = p12(k)
           name1=array_dict.keys()[0]
           name2=array_dict.keys()[1]
           bm_1=array_dict[name1]['bm']/2.
           bm2_1=array_dict[name1]['bm2']/2.
           bm_eff_1=bm_1**2./bm2_1
           bm_2=array_dict[name2]['bm']/2.
           bm2_2=array_dict[name2]['bm2']/2.
           bm_eff_2=bm_2**2./bm2_2

           Tsys1=array_dict[name1]['Tsky']+array_dict[name1]['Trx']
           Tsys2=array_dict[name2]['Tsky']+array_dict[name2]['Trx']

           
           
           scalar1 = X2Y(z,array_dict[name1]['line_freq']) * bm_eff_1 * B * k**3 / (2*n.pi**2)
           scaler2 = X2Y(z,array_dict[name2]['line_freq']) * bm_eff_2 * B * k**3 / (2*n.pi**2)
           Trms1 = Tsys1 / n.sqrt(2*(array_dict[name1]['B']*1e9)*tot_integration_1)
           Trms2 = Tsys2 / n.sqrt(2*(array_dict[name2]['B']*1e9)*tot_integration_2)
           #add errors in inverse quadrature
           if delta12==delta1 and delta2==delta12 and scalar1*Trms1**2==scaler2*Trms2:
               #if cross power is equal to auto-power, than we want the power spectrum sensitivity formula (n12=n11**2)
               #this is physically incorrect since the power-spectrum for identical quantities is usually derived through
               #interleaved visibilities. However, we get the correct formulas for cross and auto power spectra this way. 
               n12=scalar1*Trms1**2
           else:#otherwise, want to use cross-power spectrum formula
               n12=0.
           autovar=2./((scalar1*Trms1**2. + delta1)*(scaler2*Trms2**2 + delta2)+(delta12+n12))**2.
           lst_factor=(1+(lst_ratio-1)*delta12**2./((delta12+(scaler1*Trms1**2+delta1)*(scaler2*Trms2**2+delta2))))/array_dict[max_name]['n_lstbins']
           nvar=1./((scalar1*Trms1**2)*(scaler2*Trms2**2))
           sense[kpr][i] += autovar/lst_factor
           Tsense[kpr][i] += nvar/lst_factor
           corrSense[kpr][i] +=autovar*(n_corr-1)*delta12**2./((delta1+scaler1*Trms1**2.)*(delta2+scaler2*Trms2**2.))/lst_factor
           
           #multiply by lst factors
           
#multiply sense and Tsens by LST factor

#bin the result in 1D
delta = dk_deta(z,line_freq)*(1./B) #default bin size is given by bandwidth
kmag = n.arange(delta,n.max(mk),delta)

kprs = n.array(kprs)
sense1d = n.zeros_like(kmag)
Tsense1d = n.zeros_like(kmag)
for ind, kpr in enumerate(sense.keys()):
    #errors were added in inverse quadrature, now need to invert and take square root to have error bars; also divide errors by number of indep. fields
    #sense[kpr] = sense[kpr]**-.5 #/ n.sqrt(n_lstbins)
    #Tsense[kpr] = Tsense[kpr]**-.5 #/ n.sqrt(n_lstbins)
    for i, kpl in enumerate(kpls):
        k = n.sqrt(kpl**2 + kpr**2)
        delta12=p12(k)
        delta1=p1(k)
        delta2=p2(k)
        if k > n.max(mk): continue
        #add errors in inverse quadrature for further binning
        sense1d[find_nearest(kmag,k)] += sense[kpr][i]
        Tsense1d[find_nearest(kmag,k)] += Tsense[kpr][i]
        corrsense1d[find_nearest(kmag,k)]+=corrsense1d[find_nearest(kmag,k)]
#invert errors and take square root again for final answer
for ind,kbin in enumerate(sense1d):
    sense1d[ind] = kbin**-.5
    Tsense1d[ind] = Tsense1d[ind]**-.5

#save results to output npz
n.savez('%s_%s_%.3f.npz' % (name,opts.model,array['freq']),ks=kmag,errs=sense1d,T_errs=Tsense1d)

#calculate significance with least-squares fit of amplitude
A = p12(kmag)
M = p12(kmag)
wA, wM = A * (1./sense1d), M * (1./sense1d)
wA, wM = n.matrix(wA).T, n.matrix(wM).T
amp = (wA.T*wA).I * (wA.T * wM)
#errorbars
Y = n.float(amp) * wA
dY = wM - Y
s2 = (len(wM)-1)**-1 * (dY.T * dY)
X = n.matrix(wA).T * n.matrix(wA)
err = n.sqrt((1./n.float(X)))
print 'total snr = ', amp/err
