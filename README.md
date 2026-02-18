# Setup

Install the packages in `requirements.txt` in your python environment. With pip that would be - 

```
pip install -r requirements.txt
```


Change these lines at the bottom of `ompldg_astrometry.py` to your local paths


```
if __name__ == "__main__":
    catalog_path = "./OMPLDG_croin_cassan.sample.csv" # catalog
    events_directory_path = "./sample_rtmodel_v3.2_ICGS" # results from Stela's runs
    satellitedir = "./satellitedir" # Ephemeris in VBM format
    output_dir = './' #Where you want to save the files
```

# Usage

```
python3 ompldg_astrometry.py event_2_793_3156 LS0017-4.txt 
```

# Output 

```
HJD_prime mag err source_centroid_RA source_centroid_Dec lens_centroid_RA lens_centroid_Dec centroid_RA centroid_Dec
8346.501847300213 20.083626766945223 0.0069212018179713 -4.228715877512374 1.3791988510259352 -4.228715877512374 1.3791988510259352 -2.7395557380319357 5.297978105306492
8346.512077500112 20.098129521947012 0.007014272197153 -4.228645918970073 1.3791679620837916 -4.228645918970073 1.3791679620837916 -2.7395269651745853 5.297845003074556
...
```
