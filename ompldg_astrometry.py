import sys
import pandas as pd
import numpy as np
import VBMicrolensing
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

def main(catalog_path_, events_directory_path_, satellitedir_, event_name_, model_name_,output_dir_):

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
    #######################################
    # Convert Ds to piS in mas, muS_l/b to muS_RA/Dec
    ########################################
    piS = 1/(Ds*1000) * (3600) # Source Parallax
    # easiest to use astropy here
    coord_galactic = SkyCoord(l=source_l*u.deg,b=source_b*u.deg
                    ,pm_l_cosb=muS_helio_l*np.cos(source_b)*u.mas/u.yr,
                    pm_b=muS_helio_b*u.mas/u.yr,frame='galactic')
    coord_icrs = coord_galactic.icrs
    muS_helio_ra = coord_icrs.pm_ra_cosdec.value/np.cos(coord_icrs.dec.value)
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
    parameters[0] = np.log(parameters[0]) # s
    parameters[1] = np.log(parameters[1]) # q
    parameters[4] = np.log(parameters[4]) # rho
    parameters[5] = np.log(parameters[5]) # tE
    #truncate array after PLX. If model type is LS grab the PLX from the catalog and use that.
    # For now neglect orbital motion.
    if model_name_[0:2] == 'LS':
        print('Warning, you provided a static model. Using parallax from the catalog.')
        piEN = event_in_catalog.loc[0,'piEN']
        piEE = event_in_catalog.loc[0, 'piEE']
        print(f'PiEN = {piEN}, piEE = {piEE}')
        parameters = parameters[0:7]
        append_parameters = [piEN,piEE,muS_helio_dec,muS_helio_ra, piS, thetaE]
    else:
        parameters = parameters[0:9]
        append_parameters = [muS_helio_dec, muS_helio_ra, piS, thetaE]
    # add astrometric parameters to modeled parameters
    for par in append_parameters:
        parameters.append(par)
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
    results = vbm.BinaryAstroLightCurve(parameters,times)
    # Now use VBMicrolensing function to combine the centroids.
    # Now this function assumes the only source of blend flux is the lens.
    # I will have to modify the function if Dave wants to include other blending.
    # for now I will inlclude a check you can use to compare Lens/Source flux ration to the gulls blending.

    g = 10**(-0.4*(lens_F146 - source_F146))
    #fbaseline = fsource/fs
    fl = (10**(-0.4*lens_F146))/((10**(-0.4*source_F146))/fs) # lens flux fraction
    # now calculate ratio of gulls blending to source flux
    gulls_g = (1-fs)/fs
    print("Currently ignores non-lens blend. Compare lens-source flux ratio to gulls blend-source ratio.")
    print(f'Lens-Source ratio = {g}, Blend-Source ratio = {gulls_g}')
    print(f'Lens contributes {100*g/gulls_g} % of total blend flux')

    # combine the centroids
    combined_centroid = vbm.CombineCentroids(results, g)

    fig,axs = plt.subplots(1,2,figsize = (10,5),layout = 'constrained')
    ax = axs[0]
    ax.plot(times,results[0],'.')
    ax.set_xlabel('HJD - 2450000')
    ax.set_ylabel('Magnification')
    ax.set_xlim(parameters[6]-np.exp(parameters[5]),parameters[6]+np.exp(parameters[5]))

    ax = axs[1]
    source_centroid = [results[1], results[2]]
    lens_centroid = [results[3], results[4]]
    ax.plot(source_centroid[1], source_centroid[0],'.',color = 'blue' ,markersize=0.1)
    ax.plot(lens_centroid[1], lens_centroid[0],'.',color='chartreuse',markersize=0.1)
    ax.plot(combined_centroid[1], combined_centroid[0], '.',color='coral',markersize=0.1)


    ax.plot(-1000,1000,'.',label = 'source centroid',markersize=10,color = 'blue')
    ax.plot(-1000, 1000, '.', label='lens centroid', markersize=10, color='chartreuse')
    ax.plot(-1000, 1000, '.', label='combined centroid', markersize=10, color='coral')
    ax.legend()
    ran = 1
    ax.set_ylim(-ran, ran)
    ax.set_xlim(ran, -ran)
    ax.set_xlabel('dRA (mas)')
    ax.set_ylabel('dDec (mas)')
    plt.show()

    F146_data = F146_data.loc[:,['HJD_prime','mag','err']]
    F146_data['magnification'] = results[0]
    F146_data['source_centroid_RA'] = source_centroid[1]
    F146_data['source_centroid_Dec'] = source_centroid[0]
    F146_data['lens_centroid_RA'] = lens_centroid[1]
    F146_data['lens_centroid_Dec'] = lens_centroid[0]
    F146_data['centroid_RA'] = combined_centroid[1]
    F146_data['centroid_Dec'] = combined_centroid[0]

    save_path = f'{output_dir_}/{event_name_}_F146_astrometry.dat'
    F146_data.to_csv(save_path,index=False,sep=' ')










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
    events_directory_path = "./sample_rtmodel_v3.2_ICGS" # results from Stela's runs
    satellitedir = "./satellitedir" # Ephemeris in VBM format
    output_dir = './' #Where you want to save the files
    event_name = sys.argv[1]
    try:
        model_name = sys.argv[2]
    #possible to not pass the model name.
    except IndexError:
        model_name = None
        raise ValueError('For now you must provide a model name.')
    main(catalog_path_=catalog_path,events_directory_path_=events_directory_path,satellitedir_=satellitedir,
         event_name_=event_name,model_name_=model_name,output_dir_=output_dir)



