The programs found here were used in conjunction to better understand the physics of automated glider control.

SuperBIT, a conglomerate of companies including NASA, University of Toronto, and CSA, are using an innovative new balloon technology to put a telescope into the very upper atmosphere (about 35km high). Due to the unpredictability of where the telescope would finally (crash) land, novel techniques for the safe return of the data (photos taken by the telescope) are being explored. One possibility is using automated gliders to safely carry a hard drive down through the atmosphere to a predetermined landing location. This research project is a proof-of-concept of automated glider return. Read more about the SuperBIT project here: https://sites.physics.utoronto.ca/bit/

These programs all create data visualisations, for use in interpretation and as visual aids for client-centred communication.

The associated import data files have also been included, should one wish to run the program. The filepaths must be edited for your personal system.

Hopefully you will find these interesting to look at!

## PROGRAM 1 - 'Navigational_algorithm.py'
This program utilises genetic optimisation to quickly find an efficient flight path to a selected landing point. The threshold for success can be easily changed such that the glider will be expected to land within, say, 10m of the target. To a satisfactory threshold of 100m radius, the program takes an average of ~3 seconds to improve accuracy ~500% from first 'good' guess. It also incorporates weather forecast data, such that forecasts (or current conditions) on the day can be fed to the program to allow the glider to correctly navigate through the often-treacherous atmosphere.
The flight path information is then fed directly to the glider: in our case, the glider was controlled by an HKPilot32 (an Arduino I/O analogue) with an uploaded custom flight nav algorithm.
The program also plots, in 3d, the first 'good' guess flight path and the optimised path. These visuals can be seen in the two accompanying files, 'Navigational_algorithm view 1-2.png'

## PROGRAM 2 - 'Landing_model.py'
The program displays a more microscopic view of the path, dealing specifically with the landing mechanics. The glider was directed to the approximate landing area as a priority, before the final descent had begun. A slow, controlled spiral is used to minimise risk of high-speed crash, or wildly incorrect landing coordinates.
This is visualised in the accompanying figures, 'Landing_model view 1-3.png'
