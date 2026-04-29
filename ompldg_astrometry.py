import sys
import pandas as pd
import numpy as np
import VBMicrolensing
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
def main(catalog_path_, events_directory_path_, satellitedir_, event_name_, model_name_, output_dir_, plot_folder,windowsize=1, noisefloor=1.1):

    ### Set up VBMicrolensing
    coordinatefile = f'{events_directory_path_}/{event_name_}/Data/event.coordinates'
    # set coordinates and ephemerides path
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.SetObjectCoordinates(coordinatefile, satellitedir_)
    vbm.a1 = 0.36 # set limb darkening
    vbm.satellite = 1 # use W146 satellite
    #######################################
    # Read in the model information
    ########################################
    path_to_model = f'{events_directory_path_}/{event_name_}/Models/{model_name_}'
    with open(path_to_model,'r') as modelfile:
        model_string = modelfile.readline()# All relevant model parameters are in first line. read that.
    parameters = list(map(float, model_string.split(' '))) #convert it to a python list
    ########################################
    # Now get the relevant astrometry parameters from the catalog csv
    # Need Source_W146, Lens_W146, source flux fraction Obs_0_fs
    # Source_mul Source_mub (have to convert to RA Dec) Source_Dist (kpc) thetaE
    # Source_l	Source_b
    ########################################
    cat = pd.read_csv(catalog_path_)
    event_in_catalog = find_event_in_catalog(cat,event_name_)
    source_l = event_in_catalog.loc[0,'Source_l']; source_b = event_in_catalog.loc[0,'Source_b']
    muS_helio_l = event_in_catalog.loc[0,'Source_mul']; muS_helio_b = event_in_catalog.loc[0,'Source_mub']
    Ds = event_in_catalog.loc[0,'Source_Dist']; thetaE = event_in_catalog.loc[0,'thetaE']
    source_F146 = event_in_catalog.loc[0,'Source_W146']; lens_F146 = event_in_catalog.loc[0,'Lens_W146']
    fs = event_in_catalog.loc[0,'Obs_0_fs'] # source flux fraction

    input_planetary_pars = get_input_parameters_string(event_in_catalog)


    #######################################
    # Convert Ds to piS in mas, muS_l/b to muS_RA/Dec
    ########################################
    piS = 1/(Ds*1000) * (1000) # Source Parallax
    # easiest to use astropy here
    coord_galactic = SkyCoord(l=source_l*u.deg,b=source_b*u.deg
                    ,pm_l_cosb=muS_helio_l*np.cos(source_b*np.pi/180)*u.mas/u.yr,
                    pm_b=muS_helio_b*u.mas/u.yr,frame='galactic')
    coord_icrs = coord_galactic.icrs
    print('Dec',coord_icrs.dec.value)
    muS_helio_ra = coord_icrs.pm_ra_cosdec.value/np.cos(coord_icrs.dec.value*np.pi/180)
    muS_helio_dec = coord_icrs.pm_dec.value


    # some diagnostics to check the conversion. It looks OK
    #print(muS_helio_ra,muS_helio_dec)
    #print(np.sqrt((muS_helio_ra*np.cos(coord_icrs.dec.value))**2+muS_helio_dec**2))
    #print(np.sqrt((muS_helio_l*np.cos(source_b))**2 + muS_helio_b**2))
    #print(event_in_catalog.loc[0,['Source_RA20000','Source_DEC20000']])
    ####################################
    # OK now set up centroid calculation.
    ####################################
    vbm.turn_off_secondary_lens = True # not all that necessary but guarantees all lens flux comes from primary
    # RTModel reports some parameters s,q, tE rho, that VBMicrolensing wants as log()

    RTModel_parameters_for_str = parameters.copy()
    chi2 = parameters[-1]

    parameters[0] = float(np.log(parameters[0])) # s
    parameters[1] = float(np.log(parameters[1])) # q
    parameters[4] = float(np.log(parameters[4])) # rho
    parameters[5] = float(np.log(parameters[5])) # tE
    #truncate array after PLX. If model type is LS grab the PLX from the catalog and use that.
    # For now neglect orbital motion.
    if model_name_[0:2] == 'LS':
        print('Warning, you provided a static model. Using parallax from the catalog.')
        piEN = event_in_catalog.loc[0,'piEN']
        piEE = event_in_catalog.loc[0, 'piEE']
        print(f'PiEN = {piEN}, piEE = {piEE}')
        iflux = 7
        append_parameters = [piEN,piEE,muS_helio_dec,muS_helio_ra, piS, thetaE]
    elif model_name_[0:2] == 'LX':
        iflux = 9
        append_parameters = [muS_helio_dec, muS_helio_ra, piS, thetaE]
    elif model_name_[0:2] == 'LO':
        iflux = 12
        append_parameters = [muS_helio_dec, muS_helio_ra, piS, thetaE]
    else:
        print('Need to provide one of LS LX or LO')
        return None
    # add astrometric parameters to modeled parameters
    # get measured fluxes
    background_flux_measured = parameters[iflux]
    source_flux_measured = parameters[iflux+1]
    # Chop off fluxes and chi2
    parameters = parameters[0:iflux]
    RTModel_parameters_for_str = RTModel_parameters_for_str[0:iflux]
    # add the astrometric parameters
    for par in append_parameters:
        parameters.append(float(par))
        RTModel_parameters_for_str.append(float(par))
        input_planetary_pars.append(float(par))
    # add magnitudes to string arrays

    #######################################
    # Read times array in from F146 file in event.
    # TODO update to overguide 12.1 m cadence?
    # Fairly easy to do, maybe have to use somewhat simplified error model.
    # Would be below resolution of the ephemeris files but VBMicrolensing can interpolate.
    ########################################
    F146_file = f'{events_directory_path_}/{event_name_}/Data/RomanW146sat1.dat'
    F146_data = pd.read_csv(F146_file,sep = '\s+', names = ['mag','err', 'HJD_prime'], comment='#')
    times = F146_data['HJD_prime'].values
    #######################################
    # Calculate magnification and centroids
    ########################################
    nodef_parameters = parameters.copy()
    nodef_parameters[-1] /= 1000

    times2 = np.linspace(np.min(times),np.max(times),100_000)
    if model_name_[0:2] == 'LS' or model_name_[0:2] =='LX':
        results_nodef = vbm.BinaryAstroLightCurve(nodef_parameters,times)
        results = vbm.BinaryAstroLightCurve(parameters,times)
        results2 = vbm.BinaryAstroLightCurve(parameters, times2)
    else:
        results_nodef = vbm.BinaryAstroLightOrbital(nodef_parameters, times)
        results = vbm.BinaryAstroLightCurveOrbital(parameters, times)
    print('Model Paramters')
    print(parameters)
    print(results[1][0], results[2][0])
    print('--------------------------------')

    # Now use VBMicrolensing function to combine the centroids.
    # Now this function assumes the only source of blend flux is the lens.
    # I will have to modify the function if Dave wants to include other blending.
    # for now I will inlclude a check you can use to compare Lens/Source flux ration to the gulls blending.

    g = 10**(-0.4*(lens_F146 - source_F146))
    #fbaseline = fsource/fs
    fl = (10**(-0.4*lens_F146))/((10**(-0.4*source_F146))/fs) # lens flux fraction
    # now calculate ratio of gulls blending to source flux
    gulls_g = (1-fs)/fs
    #print("Currently ignores non-lens blend. Compare lens-source flux ratio to gulls blend-source ratio.")

    print("Using GULLS catalog")
    print(f'Lens-Source ratio = {g}, Blend-Source ratio = {gulls_g}')
    print(f'Lens contributes {100*g/gulls_g} % of total blend flux')
    print("-------------------------------------------------------")
    g_measured_lens = (10**(-0.4*(lens_F146)))/source_flux_measured
    blend_measured = background_flux_measured/source_flux_measured
    fs_measured = 1/(1+blend_measured)
    g_measured_total = (1 - fs_measured)/fs_measured
    print("Using Measured fluxes")
    print(f'Lens-Source ratio = {g_measured_lens}, Blend-Source ratio = {g_measured_total}')
    print(f'Lens contributes {100 * g_measured_lens / g_measured_total} % of total blend flux')
    print("-------------------------------------------------------")
    print(f'fs_gulls/fs_measured = {fs/fs_measured}')

    # combine the centroids

    #copy lens centroid from actual event
    results_nodef[3] = results[3]
    results_nodef[4] = results[4]
    combined_centroid_nodef = vbm.CombineCentroids(results_nodef, g_measured_lens)

    source_centroid_nodef = [results_nodef[1], results_nodef[2]]
    lens_centroid_nodef = [results_nodef[3], results_nodef[4]]
    ##############################################################

    combined_centroid = vbm.CombineCentroids(results, g_measured_lens)
    source_centroid = [results[1], results[2]]

    source_centroid2 = [results2[1], results2[2]]
    lens_centroid = [results[3], results[4]]
    #print(results[2][0],results[1][0])

    F146_data = F146_data.loc[:,['HJD_prime','mag','err']]
    F146_data['magnification'] = results[0]
    F146_data['model_magnitude'] = -2.5*np.log10(source_flux_measured)+ 2.5*np.log10(fs_measured) -2.5*np.log10(F146_data['magnification']*fs_measured +1-fs_measured)
    source_plus_lens_magnitudes  = -2.5 * np.log10(F146_data['magnification']*source_flux_measured +10**(-0.4*lens_F146))
    F146_data['source_centroid_RA'] = source_centroid[1]
    F146_data['source_centroid_Dec'] = source_centroid[0]
    F146_data['lens_centroid_RA'] = lens_centroid[1]
    F146_data['lens_centroid_Dec'] = lens_centroid[0]
    F146_data['true_centroid_RA'] = combined_centroid[1]
    F146_data['true_centroid_Dec'] = combined_centroid[0]

    #F146_data['cont_centroid_RA'] = source_centroid2[1]
    #F146_data['cont_centroid_Dec'] = source_centroid2[0]

    F146_data['nodef_source_centroid_RA'] = source_centroid_nodef[1]
    F146_data['nodef_source_centroid_Dec'] = source_centroid_nodef[0]
    #F146_data['lens_centroid_RA'] = lens_centroid[1]
    #F146_data['lens_centroid_Dec'] = lens_centroid[0]
    F146_data['nodef_centroid_RA'] = combined_centroid_nodef[1]
    F146_data['nodef_centroid_Dec'] = combined_centroid_nodef[0]


    # background_centroids_ra = np.zeros((F146_data.shape[0],1))
    # background_centroids_dec = np.zeros((F146_data.shape[0],1))

    #background_centroids_ra[:,0] = F146_data['source_centroid_RA']+1/(2*np.sqrt(2))
    #background_centroids_dec[:,0] = F146_data['source_centroid_Dec']+1/(2*np.sqrt(2))
    g_measured_nonlens = np.array([g_measured_total-g_measured_lens])
    print(g_measured_nonlens)
    #print(background_centroids_ra.shape,background_g.shape)
    # CombineNCentroids(results=F146_data,g=g_measured_lens,background_centroids_ra=background_centroids_ra,
    #                   background_centroids_dec=background_centroids_dec,g_background=g_measured_nonlens)

    #calculate astrometric errors and add to the data
    print(event_name_)
    print('Source Proper Motion')
    print(muS_helio_ra,muS_helio_dec)
    print('-------------------------')
    F146_data = add_astrometric_noise(F146_data,noise_floor=noisefloor,magnitudes = source_plus_lens_magnitudes)


    # Add fluxes to output string
    RTModel_parameters_for_str.append(-2.5*np.log10(source_flux_measured))
    RTModel_parameters_for_str.append(lens_F146)
    RTModel_parameters_for_str.append(fs_measured)

    input_planetary_pars.append(source_F146)
    input_planetary_pars.append(lens_F146)
    input_planetary_pars.append(fs)

    # write to file
    if noisefloor == 0:
        noise_str = '0'
    else: noise_str = str(noisefloor)
    save_path = f'{output_dir_}/{event_name_}_F146_astrometry_noisefloor{noise_str}mas.dat'
    label_str = '#type s q u0 alpha rho tE t0 piEN piEE muS_helio_dec muS_helio_ra piS thetaE source_mag lens_mag fs'
    input_str = 'catalog '
    RTModel_str='RTModel '
    with open(save_path, 'w') as (f):
        for i in range(len(input_planetary_pars)):
            input_str+= str(input_planetary_pars[i])+' '
            RTModel_str +=str(RTModel_parameters_for_str[i]) + ' '
        RTModel_str+= f'chi2 = {chi2}'
        f.write(label_str+'\n')
        f.write(input_str + '\n')
        f.write(RTModel_str + '\n')

    F146_data.to_csv(save_path,index=False,sep=' ', mode='a',na_rep='NaN')
    if type(plot_folder) is not None:
        savepath = f'{plot_folder}/{event_name_}_F146_astrometry_noisefloor{noise_str}mas.png'
        plot_centroids(F146_data,parameters,savepath = savepath,windowsize=windowsize,scont = source_centroid2)

# void VBMicrolensing::CombineCentroids(double* mags, double* c1s, double* c2s, double* c1l, double* c2l, double* c1tot, double* c2tot, double g, int np) {
# 	double fac;
# 	for (int i = 0; i < np; i++) {
# 		fac = 1 / (mags[i] + g);
# 		c1tot[i] = (c1s[i] * mags[i] + c1l[i] * g) * fac;
# 		c2tot[i] = (c2s[i] * mags[i] + c2l[i] * g) * fac;


def get_input_parameters_string(event_in_catalog):
    q = event_in_catalog.loc[0,'Planet_q']
    s = event_in_catalog.loc[0,'Planet_s']
    alpha = event_in_catalog.loc[0,'alpha']
    alpha = alpha * np.pi / 180 + np.pi
    if alpha > 2*np.pi:
        alpha -=2*np.pi
    elif alpha <2*np.pi:
        alpha +=2*np.pi
    u0_lens1 = event_in_catalog.loc[0,'u0lens1']
    t0_lens1 = event_in_catalog.loc[0,'t0lens1']+8234#put to
    tE_ref = event_in_catalog.loc[0,'tE_ref']
    rho = event_in_catalog.loc[0,'rho']
    piEE = event_in_catalog.loc[0,'piEE']; piEN = event_in_catalog.loc[0,'piEN']

    input_pars = [s,q,u0_lens1,alpha,rho,tE_ref,t0_lens1,piEN, piEE]
    return input_pars


def compute_undeflected_source_trajectory(results, pars):

    #Follows almost exactly V. Bozza's VBNIcrolensing ComputeCentroids function.

    dPosAng = 0
    thetaE = pars[-1]
    alpha = pars[3]
    piEN = pars[7]
    piEE = pars[8]
    c1s = np.array(results[5])
    c2s = np.array(results[6])
    c1 = c1s[0] * thetaE  # Centroid coordinates come in x1,x2 system of the lens
    c2 = c2s[0] * thetaE

    PosAng = np.atan2(piEN,
                      piEE) - alpha + dPosAng  # Angle between North and axis x1 of the lens system counterclockwise

    c1prov = c1 * np.cos(PosAng) - c2 * np.sin(PosAng)
    c2 = c1 * np.sin(PosAng) + c2 * np.cos(PosAng)
    c1 = c1prov  # Now centroid coordinates are in North-East system, but still relative to lens


    c1s = c1 + np.array(results[3])
    c2s = c2 + np.array(results[4])

    return c1s, c2s


def CombineNCentroids(results,g, background_centroids_ra,background_centroids_dec,g_background):
    if type(g_background)!=np.ndarray:
        g_background = np.array(g_background)

    weighted_source_lens_centroids_ra = results.loc[:, ['source_centroid_RA', 'lens_centroid_RA']].values
    weighted_source_lens_centroids_ra[:, 0] *= results.loc[:, 'magnification']
    weighted_source_lens_centroids_ra[:, 1] *= g

    weighted_source_lens_centroids_dec = results.loc[:, ['source_centroid_Dec', 'lens_centroid_Dec']].values
    weighted_source_lens_centroids_dec[:, 0] *= results.loc[:, 'magnification']
    weighted_source_lens_centroids_dec[:, 1] *= g

    weighted_centroids_ra = np.concatenate([weighted_source_lens_centroids_ra, background_centroids_ra * g_background], axis=1)
    weighted_centroids_dec = np.concatenate([weighted_source_lens_centroids_dec, background_centroids_dec * g_background],
                                           axis=1)
    fac_array = 1 / (results['magnification'].values + g + np.sum(g_background))

    results['true_centroid_RA'] = fac_array * np.sum(weighted_centroids_ra, axis=1)
    results['true_centroid_Dec'] = fac_array * np.sum(weighted_centroids_dec, axis=1)

    return results
def interpolate_astrometric_error(magnitudes,path_to_lam_errors = './lam_astro_precision.txt' , noise_floor = 1.1):
    #Uses calculations from Casey Lam based on McKinnon paper
    err_array = np.loadtxt(path_to_lam_errors)
    interpolator = interp1d(err_array[:, 0], err_array[:, 1],fill_value='extrapolate')
    errors = interpolator(magnitudes)
    # add noise floor in quadrature
    # 1.1 mas is 1% of a Roman pixel
    errors = np.sqrt(errors**2+noise_floor**2)
    return errors

def add_astrometric_noise(F146_data,noise_floor,magnitudes):
    # add astrometric noise to centroid
    astrometric_errors = interpolate_astrometric_error(magnitudes,noise_floor=noise_floor)
    measured_centroid_RA = np.random.normal(loc = F146_data['true_centroid_RA'], scale=astrometric_errors)
    measured_centroid_Dec = np.random.normal(loc=F146_data['true_centroid_Dec'], scale=astrometric_errors)
    F146_data['measured_centroid_RA'] = measured_centroid_RA
    F146_data['measured_centroid_Dec'] = measured_centroid_Dec
    F146_data['measured_centroid_RA_error'] = astrometric_errors
    F146_data['measured_centroid_Dec_error'] = astrometric_errors
    return F146_data


def plot_centroids(F146_data,parameters, savepath, windowsize = 1,scont=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout='constrained', dpi=100)
    ax = axs[0]
    ax.plot(F146_data['HJD_prime'], F146_data['mag'], '.', label='GULLS output')
    ax.plot(F146_data['HJD_prime'], F146_data['model_magnitude'], '.',label = 'Calculated from model')
    ax.legend()
    ax.invert_yaxis()
    ax.set_xlabel('HJD - 2450000')
    ax.set_ylabel('Magnification')
    ax.set_xlim(parameters[6] - np.exp(parameters[5]), parameters[6] + np.exp(parameters[5]))

    ax = axs[1]
    #ax.errorbar(F146_data['measured_centroid_RA'],F146_data['measured_centroid_Dec'],xerr = F146_data['measured_centroid_RA_error'],
    #yerr = F146_data['measured_centroid_Dec_error'], color='lime', markersize=0.1,marker='.',zorder=0,linewidth=0, elinewidth=0.1)
    #F146_data['cont_centroid_RA'] = source_centroid2[1]
    #F146_data['cont_centroid_Dec'] = source_centroid2[0]
    if scont is not None:
        ax.plot(scont[1], scont[0], '.', color='skyblue',
            markersize=0.1, zorder=10)
    ax.plot(F146_data['source_centroid_RA'], F146_data['source_centroid_Dec'], '.', color='blue',
            markersize=0.1, zorder=10)
    #ax.plot(F146_data['nodef_source_centroid_RA'], F146_data['nodef_source_centroid_Dec'], '.', color='blue', markersize=0.1,zorder=10)
    ax.plot(F146_data['lens_centroid_RA'], F146_data['lens_centroid_Dec'], '.', color='black', markersize=0.1,zorder=10)
    #ax.plot(F146_data['lens_source_centroid_RA'], F146_data['lens_source_centroid_Dec'], '.', color='black', markersize=0.1,zorder=10)
    ax.plot(F146_data['true_centroid_RA'],F146_data['true_centroid_Dec'], '.', color='coral', markersize=0.1,zorder=10)

    #ax.plot(-1000, 1000, '.', label='measured centroid', markersize=10, color='lime')
    #ax.plot(-1000, 1000, '--', label='source centroid', markersize=10, color='red')
    #ax.plot(-1000, 1000, '--', label='non-deflected source centroid', markersize=10, color='blue')
    ax.plot(-1000, 1000, '--', label='lens centroid', markersize=10, color='black')
    #ax.plot(-1000, 1000, '--', label='lens source centroid', markersize=10, color='black')
    ax.plot(-1000, 1000, '--', label='combined centroid', markersize=10, color='coral')



    ax.legend()
    ran = windowsize
    ax.set_ylim(-ran, ran)
    ax.set_xlim(ran, -ran)
    ax.set_xlabel('dRA (mas)')
    ax.set_ylabel('dDec (mas)')


    # ax = axs[1]
    # deflection_RA = F146_data['source_centroid_RA'] - F146_data['nodef_source_centroid_RA']
    # deflection_Dec = F146_data['source_centroid_Dec'] - F146_data['nodef_source_centroid_Dec']
    # ax.plot(deflection_RA, deflection_Dec, '.', color='black',
    #         markersize=0.1, zorder=10)
    # ax.plot(-1000, 1000, '--', label='Source Deflection', markersize=10, color='black')
    # ax.set_ylim(-ran, ran)
    # ax.set_xlim(ran, -ran)
    # ax.legend()
    plt.savefig(savepath)
    plt.show()

def find_event_in_catalog(catalog,event_name_):
    '''
    :param catalog:
    :param event_name_:
    :return: event_in_catalog
    '''
    event_names = catalog['lcname'].apply(lambda x: 'event_' + x[40:-7])
    event_in_catalog = catalog[event_names == event_name_]
    return event_in_catalog.reset_index(drop=True)


if __name__ == "__main__":
    catalog_path = "./OMPLDG_croin_cassan.sample.csv" # catalog

    events_directory_path = "/Users/jmbrashe/VBBOrbital/AstrometryGenerator/sample_rtmodel_v3.2_ICGS"
        #"/Users/jmbrashe/Downloads/sample_rtmodel_v2.4"
    #events_directory_path = "./sample_rtmodel_v3.2" # results from Stela's runs
    satellitedir = "./satellitedir" # Ephemeris in VBM format
    output_dir = sys.argv[1] #Where you want to save the files
    plot_folder = sys.argv[2]
    noise_floor = float(sys.argv[3])
    make_plot = True
    ran = 30

    event_names = ['0_874_19']
        #['0_129_1096', '0_161_1727', '0_638_2805', '0_641_1086', '0_920_1857', '0_955_567']
   # ['0_955_165', '0_164_1588', '0_601_2068', '0_640_1938', '1_755_2163', '0_917_59', '0_10_2937', '0_1002_785',
             #'0_7_1131', '2_1001_1542', '0_837_1827']
    model_names = ['LX0013-1.txt']

        #['LXmyfit-2.txt','LXmyfit-0.txt','LXmyfit-0.txt','LXmyfit-0.txt','LXmyfit-3.txt','LXmyfit-1.txt']
    #['LXmyfit-0.txt', 'LXmyfit-0.txt', 'LXmyfit-1.txt', 'LXmyfit-1.txt', 'LXmyfit-0.txt', 'LXmyfit-3.txt',
                  #'LXmyfit-0.txt', 'LXmyfit-0.txt', 'LXmyfit-1.txt', 'LXmyfit-0.txt', 'LXmyfit-0.txt']
    for i in range(len(event_names)):
        main(catalog_path_=catalog_path,events_directory_path_=events_directory_path,satellitedir_=satellitedir,
         event_name_='event_'+event_names[i],model_name_=model_names[i],output_dir_=output_dir,plot_folder =plot_folder,windowsize=ran,noisefloor = noise_floor)



