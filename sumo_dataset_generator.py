
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def generate_routefile():
    random.seed(84)  # make tests reproducible

    N = 22500  # number of time steps


    with open("../data/demo.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="2.5" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="50" \
guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

        <route id="EtoW" edges="EW1 EW2" />
        <route id="WtoE" edges="WE1 WE2" />
        <route id="NtoS" edges="NS1 NS2" />
        <route id="StoN" edges="SN1 SN2" />
        <route id="EtoN" edges="EW1 SN2" />
        <route id="EtoS" edges="EW1 NS2" />
        <route id="WtoN" edges="WE1 SN2" />
        <route id="WtoS" edges="WE1 NS2" />
        <route id="NtoE" edges="NS1 WE2" />
        <route id="NtoW" edges="NS1 EW2" />
        <route id="StoE" edges="SN1 WE2" />
        <route id="StoW" edges="SN1 EW2" />
        """, file=routes)

        a = 0  # alt sınır
        b = 100 # üst sınır
        vehNr = 0
        for i in range(N):

            if i < 2500:
                pEW = pEN = pES = 1. / 5
                pNS = pNE = pNW = 1. / 40
                pSN = pSE = pSW = 1. / 40
                pWS = pWN = pWE = 1. / 40

            elif 2500 < i < 3000:
                pEW = pEN = pES = 1. / 70
                pNS = pNE = pNW = 1. / 70
                pSN = pSE = pSW = 1. / 70
                pWS = pWN = pWE = 1. / 70

            elif 3000 < i < 5000:
                pEW = pEN = pES = 1. / 40
                pNS = pNE = pNW = 1. / 5
                pSN = pSE = pSW = 1. / 40
                pWS = pWN = pWE = 1. / 40

            elif 5000 < i < 5500:
                pEW = pEN = pES = 1. / 70
                pNS = pNE = pNW = 1. / 70
                pSN = pSE = pSW = 1. / 70
                pWS = pWN = pWE = 1. / 70

            elif 5500 < i < 7500:
                pEW = pEN = pES = 1. / 40
                pNS = pNE = pNW = 1. / 40
                pSN = pSE = pSW = 1. / 5
                pWS = pWN = pWE = 1. / 40

            elif 7500 < i < 8000:
                pEW = pEN = pES = 1. / 70
                pNS = pNE = pNW = 1. / 70
                pSN = pSE = pSW = 1. / 70
                pWS = pWN = pWE = 1. / 70

            elif 8000 < i < 10000:
                pEW = pEN = pES = 1. / 40
                pNS = pNE = pNW = 1. / 40
                pSN = pSE = pSW = 1. / 40
                pWS = pWN = pWE = 1. / 5

            elif 10000 < i < 10500:
                pEW = pEN = pES = 1. / 70
                pNS = pNE = pNW = 1. / 70
                pSN = pSE = pSW = 1. / 70
                pWS = pWN = pWE = 1. / 70

            elif 10500 < i < 12500:
                pEW = pEN = pES = 1. / 25
                pNS = pNE = pNW = 1. / 25
                pSN = pSE = pSW = 1. / 25
                pWS = pWN = pWE = 1. / 25

            elif 12500 < i < 13000:
                pEW = pEN = pES = 1. / 70
                pNS = pNE = pNW = 1. / 70
                pSN = pSE = pSW = 1. / 70
                pWS = pWN = pWE = 1. / 70

            elif 13000 < i < 15000:
                pEW = pEN = pES = 1. / 40
                pNS = pNE = pNW = 1. / 40
                pSN = pSE = pSW = 1. / 40
                pWS = pWN = pWE = 1. / 40

            elif 15000 < i < 15500:
                pEW = pEN = pES = 1. / 70
                pNS = pNE = pNW = 1. / 70
                pSN = pSE = pSW = 1. / 70
                pWS = pWN = pWE = 1. / 70

            elif 15500 < i < 17500:
                pEW = pEN = pES = 1. / 5
                pNS = pNE = pNW = 1. / 5
                pSN = pSE = pSW = 1. / 5
                pWS = pWN = pWE = 1. / 5

            elif 17500 < i < 20000:
                pEW = pEN = pES = 1. / 70
                pNS = pNE = pNW = 1. / 70
                pSN = pSE = pSW = 1. / 70
                pWS = pWN = pWE = 1. / 70

            elif 20000 < i < 22500:
                pEW = pEN = pES = 1. / random.uniform(a, b)
                pNS = pNE = pNW = 1. / random.uniform(a, b)
                pSN = pSE = pSW = 1. / random.uniform(a, b)
                pWS = pWN = pWE = 1. / random.uniform(a, b)

            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="WE_%i" type="typeWE" route="WtoE" depart="%i" color="1,0,1"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="EW_%i" type="typeWE" route="EtoW" depart="%i" color="1,0,0" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="NS_%i" type="typeWE" route="NtoS" depart="%i" color="1,1,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSN:
                print('    <vehicle id="SN_%i" type="typeWE" route="StoN" depart="%i" color="0,1,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEN:
                print('    <vehicle id="EN_%i" type="typeWE" route="EtoN" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pES:
                print('    <vehicle id="ES_%i" type="typeWE" route="EtoS" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNE:
                print('    <vehicle id="NE_%i" type="typeWE" route="NtoE" depart="%i" color="1,1,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNW:
                print('    <vehicle id="NW_%i" type="typeWE" route="NtoW" depart="%i" color="1,1,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSE:
                print('    <vehicle id="SE_%i" type="typeWE" route="StoE" depart="%i" color="0,1,0" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSW:
                print('    <vehicle id="SWt_%i" type="typeWE" route="StoW" depart="%i" color="0,1,0" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWS:
                print('    <vehicle id="WS_%i" type="typeWE" route="WtoS" depart="%i" color="1,0,1"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWN:
                print('    <vehicle id="WN_%i" type="typeWE" route="WtoN" depart="%i" color="1,0,1"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)


# this is the main entry point of this script
if __name__ == "__main__":

    generate_routefile()

