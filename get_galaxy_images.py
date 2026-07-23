import matplotlib.pyplot as plt
import numpy as np
import os
from time import sleep, time

import astropy.io.fits as fits 
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.visualization import astropy_mpl_style, ImageNormalize, PercentileInterval, LogStretch, LinearStretch, AsinhStretch, SqrtStretch, PowerStretch
import astropy.units as u

from astroquery.esa.euclid import Euclid

def sec_to_hms(sec):
   sec = sec % (24 * 3600)
   hour = sec // 3600
   sec %= 3600
   min = sec // 60
   sec %= 60
   return "%02d:%02d:%02d" % (hour, min, sec) 

start_time = time()

current_path = os.getcwd()
gal_merger_filename=current_path+'/Q1_merger_classification.fits'
gal_merger = Table.read(gal_merger_filename, format='fits', hdu=1)

normal_mask = [
    (gal_merger['CNN classification'] == 0) &
    (gal_merger['CNN pred'] < 0.10)
][0]
anomalous_mask = [
    (gal_merger['CNN classification'] == 1) &
    (gal_merger['CNN pred'] > 0.90)
][0]

unknown_mask = [
    (gal_merger['CNN classification'] == -99)
][0]

normal_galaxies = gal_merger[normal_mask]
anomalous_galaxies = gal_merger[anomalous_mask]
unknown_galaxies = gal_merger[unknown_mask]

print(f"{len(normal_galaxies)} normal galaxies found of {len(gal_merger)} total!")
print(f"{len(anomalous_galaxies)} anomalous galaxies found of {len(gal_merger)} total!")
print(f"{len(unknown_galaxies)} unknown galaxies found of {len(gal_merger)} total!")



search_radius = 0.5/60
count = 1
cutouts = []

for gal in anomalous_galaxies:
    try:
        print(gal['CNN pred'], gal['CNN classification'])

        ra = gal['right_ascension']
        dec = gal['declination']

        if not os.path.isfile(f'anomalous galaxies/galaxy_{str(ra).replace(".", "_")}_{str(dec).replace(".", "_")}.png'):
            print(f"Getting anomalous galaxy #{count} @ RA: {ra*u.deg}, Dec: {dec*u.deg}")
            
            coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            query = f"""
            SELECT *
            FROM q1.mosaic_product
            WHERE instrument_name='VIS' AND 
                INTERSECTS(CIRCLE({ra}, {dec}, {search_radius}), fov)=1   
            """

            res = Euclid.launch_job_async(query).get_results()
            file_path  = f"{res['file_path'][0]}/{res['file_name'][0]}"
            
            cutout_out = Euclid.get_cutout(file_path=file_path, coordinate=coords, radius=4*u.arcsec, output_file='gal.fits')
            cutouts.append(cutout_out)

            hdul  = fits.open(cutout_out[0])
            image_data = hdul[0].data
        
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            
            plt.imshow(image_data, cmap='gray', origin='lower', norm=ImageNormalize(image_data, interval=PercentileInterval(98), stretch=LinearStretch()))
            plt.savefig(f'anomalous galaxies/galaxy_{str(ra).replace(".", "_")}_{str(dec).replace(".", "_")}.png', bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

            print(f"Galaxy #{count} saved to anomalous galaxies/galaxy_{str(ra).replace(".", "_")}_{str(dec).replace(".", "_")}.png!")

            sleep(2 + np.random.random_sample(1)[0])  # Sleep for 1 second to avoid overwhelming the server
            count += 1
    except Exception as e:
        print(f"Error getting galaxy #{count} @ RA: {ra*u.deg}, Dec: {dec*u.deg}: {e}")
        print("Sleeping for 2 minutes before continuing...")
        sleep(120)  # Sleep for 90 seconds to avoid overwhelming the server
        continue
    
end_time = time()
elapsed_time = end_time - start_time
print(f"Total time taken to save all images: {sec_to_hms(elapsed_time)}")

hdul  = fits.open(cutouts[0][0])
image_data = hdul[0].data

plt.figure()
plt.title("Galaxy #1")

plt.imshow(image_data, cmap='gray', origin='lower', norm=ImageNormalize(image_data, interval=PercentileInterval(98), stretch=LinearStretch()))
colorbar = plt.colorbar()

hdul.close()
plt.show()

