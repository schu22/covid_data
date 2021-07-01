# All COVID-19 data from January 20th to July 25th


# LOADS MAIN DATA NECESSARY FOR PROGRAM TO WORK

# Loading necessary libraries/modules

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from math import fabs


# Loads infection data from text file

infectiondata = np.genfromtxt("time_series_covid19_confirmed_US5.txt", dtype="str")
l1 = len(infectiondata)
w1 = len(infectiondata[0])
justinfectiondata = infectiondata[1:l1]
infectionheading = list(infectiondata[0])
infectiondatabetter = []
for i in range(l1):
    for j in range(w1):
        if "_" in infectiondata[i][j]:
            infectiondatabetter.append(infectiondata[i][j].replace("_", " "))
        else:
            infectiondatabetter.append(infectiondata[i][j])
infectiondatabetter = np.array(infectiondatabetter)
infectiondatabetter = np.reshape(infectiondatabetter, (l1, w1))


# Loads death data from text file

deathdata = np.genfromtxt("time_series_covid19_deaths_US4.txt", dtype="str")
l2 = len(deathdata)
w2 = len(deathdata[0])
justdeathdata = deathdata[1:l2]
deathheading = list(deathdata[0])
deathdatabetter = []
for i in range(l2):
    for j in range(w2):
        if "_" in deathdata[i][j]:
            deathdatabetter.append(deathdata[i][j].replace("_", " "))
        else:
            deathdatabetter.append(deathdata[i][j])
deathdatabetter = np.array(deathdatabetter)
deathdatabetter = np.reshape(deathdatabetter, (l2, w2))


# Defines common constants I use throughout the program

placeindex = deathheading.index('Province_State')
isindex = infectionheading.index('1')
dsindex = deathheading.index('1')
popindex = deathheading.index('Population')
numdays = len(infectionheading) - dsindex
marg = 0.5


# CREATES LISTS FOR ALL THE PLACES THAT CAN BE USED IN THIS PROGRAM

# All places– states, provinces, DC, and cruises

places = []
for i in range(1, l1):
    if infectiondatabetter[i][placeindex] not in places:
        places.append(infectiondatabetter[i][placeindex])
totalpopulation = 0
for i in range(1, l2):
    totalpopulation += int(deathdatabetter[i][popindex])
grandinfectiontotal = [0] * numdays
for i in range(1, l1):
    for j in range(numdays):
        grandinfectiontotal[j] += int(infectiondatabetter[i][j+isindex])
granddeathtotal = [0] * numdays
for i in range(1, l2):
    for j in range(numdays):
        granddeathtotal[j] += int(deathdatabetter[i][j+dsindex])

# States, provinces, and DC

statesandprovinces = places[0:len(places) - 2]
sandppop = 0
for i in range(1, l2):
    if deathdatabetter[i][placeindex] in statesandprovinces:
        sandppop += int(deathdatabetter[i][popindex])
sandpinfectiontotal = [0] * numdays
for i in range(1, l1):
    for j in range(numdays):
        if infectiondatabetter[i][placeindex] in statesandprovinces:
            sandpinfectiontotal[j] += int(infectiondatabetter[i][j+isindex])
sandpdeathtotal = [0] * numdays
for i in range(1, l2):
    for j in range(numdays):
        if infectiondatabetter[i][placeindex] in statesandprovinces:
            sandpdeathtotal[j] += int(deathdatabetter[i][j+dsindex])


# States and DC

statesanddc = places[5:len(places) - 2]
sanddcpop = 0
for i in range(1, l2):
    if deathdatabetter[i][placeindex] in statesanddc:
        sanddcpop += int(deathdatabetter[i][popindex])
sanddcinfectiontotal = [0] * numdays
for i in range(1, l1):
    for j in range(numdays):
        if infectiondatabetter[i][placeindex] in statesanddc:
            sanddcinfectiontotal[j] += int(infectiondatabetter[i][j+isindex])
sanddcdeathtotal = [0] * numdays
for i in range(1, l2):
    for j in range(numdays):
        if deathdatabetter[i][placeindex] in statesanddc:
            sanddcdeathtotal[j] += int(deathdatabetter[i][j+dsindex])


# States

states = statesanddc[0:8] + statesanddc[9:51]
statespop = 0
for i in range(1, l2):
    if deathdatabetter[i][placeindex] in states:
        statespop += int(deathdatabetter[i][popindex])
statesinfectiontotal = [0] * numdays
for i in range(1, l1):
    for j in range(numdays):
        if infectiondatabetter[i][placeindex] in states:
            statesinfectiontotal[j] += int(infectiondatabetter[i][j+isindex])
statesdeathtotal = [0] * numdays
for i in range(1, l2):
    for j in range(numdays):
        if deathdatabetter[i][placeindex] in states:
            statesdeathtotal[j] += int(deathdatabetter[i][j+dsindex])

# Takes data on the governors of states, creates lists based on that and a program to say what party the governor
# of any one state belongs to
# As of July 31, 2020

# States with Republican governors

Republican_Governor = ["Alabama", "Alaska", "Arizona", "Arkansas", "Florida",
                       "Georgia", "Idaho", "Indiana", "Iowa", "Maryland",
                       "Massachusetts", "Mississippi", "Missouri", "Nebraska",
                       "New Hampshire", "North Dakota", "Ohio", "Oklahoma",
                       "South Carolina", "South Dakota", "Tennessee",
                       "Texas", "Utah", "Vermont", "West Virginia", "Wyoming"]
gopinfectiontotal = [0] * numdays
for i in range(1, l1):
    for j in range(numdays):
        if infectiondatabetter[i][placeindex] in Republican_Governor:
            gopinfectiontotal[j] += int(infectiondatabetter[i][j+isindex])
goppopulation = 0
gopdeathtotal = [0] * numdays
for i in range(1, l2):
    for j in range(numdays):
        if deathdatabetter[i][placeindex] in Republican_Governor:
            gopdeathtotal[j] += int(deathdatabetter[i][j+dsindex])
            goppopulation += int(deathdatabetter[i][popindex])


# States with Democratic governors

Democratic_Governor = []
for state in states:
    if state not in Republican_Governor:
        Democratic_Governor.append(state)
deminfectiontotal = [0] * numdays
for i in range(1, l1):
    for j in range(numdays):
        if infectiondatabetter[i][placeindex] in Democratic_Governor:
            deminfectiontotal[j] += int(infectiondatabetter[i][j+isindex])
dempopulation = 0
demdeathtotal = [0] * numdays
for i in range(1, l2):
    for j in range(numdays):
        if deathdatabetter[i][placeindex] in Democratic_Governor:
            demdeathtotal[j] += int(deathdatabetter[i][j+dsindex])
            dempopulation += int(deathdatabetter[i][popindex])


# COMMON FUNCTIONS I USE THROUGHOUT THE PROGRAM

def bigtitleall(lst):
    """
    Given a list, determines appropriate title for graph
    """
    if lst == grandinfectiontotal or lst == granddeathtotal:
        title = "All Places"
    elif lst == sandpinfectiontotal or lst == sandpdeathtotal:
        title = "All States and Provinces"
    elif lst == sanddcinfectiontotal or lst == sanddcdeathtotal:
        title = "All States and DC"
    elif lst == statesinfectiontotal or lst == statesdeathtotal:
        title = "All States"
    elif lst == gopinfectiontotal or lst == gopdeathtotal:
        title = "Republican-governed States"
    elif lst == deminfectiontotal or lst == demdeathtotal:
        title = "Democrat-governed States"
    else:
        title = ""
    return title


# Helper function for lag

def firstnonzero(lst):
    """
    Given a list of numbers, returns index of first nonzero value
    """
    j = 0
    while True:
        if j >= len(lst):
            return "This list has no nonzero numbers"
        elif lst[j] != 0:
            break
        else:
            j += 1
            continue
    return j


# Calculates time lag between two functions in a simplified way

def lag(lst1, lst2, margin):
    """
    Given two lists of numbers, finds the "lag" between them
    (Intended to be used with infection and death data for a given place or group of places)
    """
    # Normalizes data
    m1 = max(lst1)
    m2 = max(lst2)
    if m2 == 0 or m1 == 0:
        return "Cannot find lag"
    m = len(lst1)
    n = len(lst2)
    lst3 = []
    for j in range(n):
        lst3.append(round(m1/m2*lst2[j]))
    # Searches two lists for close values
    # If the values are within the acceptable margin,
    # finds difference between their indices
    # Adds this difference to a total sum
    # And adds one to the count of differences
    s = 0
    avg = 0
    count = 0
    for i in range(m):
        for j in range(n):
            if fabs(lst1[i]-lst2[j]) < margin:
                s += fabs(i - j)
                count += 1
    # If there is at least one difference, averages them
    if count > 0:
        avg = s / count
        return round(avg)
    # Else, finds difference between first nonzero values
    else:
        try:
            return fabs(firstnonzero(lst1) - firstnonzero(lst2))
        except ValueError:
            return "Cannot find lag"


# Returns population of a place or a list of places

def population(place):
    """
    Given set lists defined earlier or a place/state, returns population
    """
    # Defines population for set lists
    if place in [grandinfectiontotal, granddeathtotal]:
        return totalpopulation
    elif place in [sandpinfectiontotal, sandpdeathtotal]:
        return sandppop
    elif place in [sanddcinfectiontotal, sanddcdeathtotal]:
        return sanddcpop
    elif place in [statesinfectiontotal, statesdeathtotal]:
        return statespop
    elif place in [gopinfectiontotal, gopdeathtotal]:
        return goppopulation
    elif place in [deminfectiontotal, demdeathtotal]:
        return dempopulation
    # Scans array for place, if area in place, adds its population to total
    else:
        pop = 0
        for row in deathdatabetter:
            if place in row[placeindex]:
                pop += int(row[popindex])
        return pop


# Returns what party a state's governor belongs to

def whatgovernor(state):
    """
    Given a state, returns governor's political party
    """
    # Scans governor arrays for state, if it is not found, returns message saying so
    if state in Republican_Governor:
        return f"{state} has a Republican governor."
    elif state in Democratic_Governor:
        return f"{state} has a Democratic governor."
    else:
        return f"Information on {state}'s leadership not available."


# FUNCTIONS THAT CREATE LISTS OR GRAPHS

# Given any state/province/DC/cruise ship, finds total number of cases and daily growth

def oneplace(whichone, place, percapita=False, k=1):
    """
    Given place and other parameters, returns total number of cases/deaths and daily growth
        whichone: "infection", "death", "both", determines what array(s) to draw data from
        place: what place to analyze
        percapita: whether data is per capita or in absolute numbers
        k: number of days to take moving average over
    """
    # Determines different cases based on what you are analyzing
    whichone = whichone.lower()
    if whichone == "infection":
        x = infectiondatabetter
        y = isindex
    elif whichone == "death":
        x = deathdatabetter
        y = dsindex
    elif whichone == "both":
        iresults = oneplace("infection", place, percapita, k)
        dresults = oneplace("death", place, percapita, k)
        if k == 1:
            ilst, itotal, idiff, iabstot = iresults
            dlst, dtotal, ddiff, dabstot = dresults
            return ilst, itotal, idiff, dlst, dtotal, ddiff, iabstot, dabstot
        else:
            ilst, itotal, itotalavg, idiff, idifavg, iabstot = iresults
            dlst, dtotal, dtotalavg, ddiff, ddifavg, dabstot = dresults
            return ilst, itotal, itotalavg, idiff, idifavg, dlst, dtotal, dtotalavg, ddiff, ddifavg, iabstot, dabstot
    else:
        return "Must input one of the following: infection, death, both"
    # Creates lists of indices in data array, absolute totals
    # (and per capita totals if applicable), daily growths
    pop = population(place)
    lst = []
    for i in range(l1):
        if place in x[i]:
            lst.append(i)
    absolutetotal = [0] * numdays
    total = [0] * numdays
    for i in lst:
        for j in range(numdays):
            absolutetotal[j] += int(x[i][j+y])
    if percapita == True:
        for i in range(numdays):
            total[i] = absolutetotal[i] / pop
    else:
        total = absolutetotal
    differences = []
    for i in range(1, numdays):
        differences.append(total[i]-total[i-1])
    # If k is not 1 (meaning you are taking a moving average of the total and growth),
    # takes averages and returns them as well
    if k != 1:
        totalaverage = [0] * numdays
        totalaverage[0] = total[0]
        for i in range(1, k-1):
            totalaverage[i] = mean(total[0:i])
        for i in range(k-1, numdays):
            totalaverage[i] = mean(total[i+1-k:i])
        diffaverage = [0] * numdays
        diffaverage[0] = differences[0]
        for i in range(1, k-1):
            diffaverage[i] = mean(differences[0:i])
        for i in range(k-1, numdays):
            diffaverage[i] = mean(differences[i+1-k:i+1])
        return lst, total, totalaverage, differences, diffaverage, absolutetotal
    else:
        return lst, total, differences, absolutetotal


# Given any state/province/DC/cruise ship, graphs total number of cases and daily growth

def graphplace(whichone, place, percapita=False, k=1, graphindex=False, yscale="Linear"):
    """
    Given place and other parameters, returns total number of cases and daily growth
        whichone: "infection", "death", "both", determines what array(s) to draw data from
        place: what place to analyze
        percapita: whether data is per capita or in absolute numbers
        k: number of days to take moving average over
        graphindex: whether to graph index of maximum daily growth
        yscale: scale of y-axis (linear, logarithmic)
    """
    # Setting up variables, prints governor's political party
    whichone = whichone.lower()
    print(whatgovernor(place))
    yscale = yscale.lower()
    # Graphing one thing: infections or deaths
    if whichone in ["infection", "death"]:
        placeresults = oneplace(whichone, place, percapita, k)
        if k == 1:
            print(place)
            total = placeresults[1]
            differences = placeresults[2]
        else:
            print(f"{place}'s {k}-day average")
            total = placeresults[2]
            differences = placeresults[4]
        maxindex = differences.index(max(differences))
        # Graph
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
        # Total cases/deaths
        if whichone == "infection":
            ax1.set_title("Confirmed cases")
        if whichone == "death":
            ax1.set_title("Deaths")
        ax1.set_xlim(0, len(total))
        ax1.set_xlabel('days since 1/25')
        ax1.plot(total, color='b')
        # Daily growth
        ax2.set_title("Daily growth")
        ax2.set_xlim(0, len(differences))
        ax1.set_xlabel('days since 1/25')
        ax2.plot(differences, color='g')
        # If graphindex is true, graphs index of maximum daily growth
        if graphindex == True:
            ax1.axvline(x=maxindex, color='y', linestyle='--')
            ax2.axvline(x=maxindex, color='y', linestyle='--')
        f.tight_layout()
    # Graphing both infections and deaths
    elif whichone == "both":
        print("Both infections and deaths")
        results = oneplace(whichone, place, percapita, k)
        if k == 1:
            print(place)
            itotal, idiff, dtotal, ddiff, iabstot, dabstot = results[1], results[2], results[4], results[5], results[6], results[7]
        else:
            print(f"{place}'s {k}-day average")
            itotal, idiff, dtotal, ddiff, iabstot, dabstot = results[2], results[4], results[7], results[9], results[10], results[11]
        imaxindex = idiff.index(max(idiff))
        dmaxindex = ddiff.index(max(ddiff))
        # Linear y-axis scale
        if yscale == "linear":
            f, (ax1, ax3) = plt.subplots(1, 2, figsize=(12.8, 4.8))
            # Plots both infection and death on graph but with different y-axes
            ax1.set_title("Total")
            ax1.set_xlabel('days since 1/25')
            ax1.set_xlim(0, len(itotal))
            ax1.set_ylabel('infection', color='r')
            ax1.set_yscale(yscale)
            ax1.tick_params(axis='y', labelcolor='r')
            ax1.plot(itotal, color='r')
            ax2 = ax1.twinx()
            ax2.set_ylabel('death', color='b')
            ax2.set_yscale(yscale)
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.plot(dtotal, color='b')
            # Plots both infection and death growths on graph but with different y-axes
            ax3.set_title("Daily growth")
            ax3.set_xlabel('days since 1/25')
            ax3.set_xlim(0, len(idiff))
            ax3.set_ylabel('infection', color='r')
            ax3.set_yscale(yscale)
            ax3.tick_params(axis='y', labelcolor='r')
            ax3.plot(idiff, color='r')
            ax4 = ax3.twinx()
            ax4.set_ylabel('death', color='b')
            ax4.set_yscale(yscale)
            ax4.tick_params(axis='y', labelcolor='b')
            ax4.plot(ddiff, color='b')
            # If graphindex is true, graphs index of maximum daily growth
            if graphindex == True:
                ax1.axvline(x=imaxindex, color='y', linestyle='--')
                ax1.axvline(x=dmaxindex, color='m', linestyle='--')
                ax2.axvline(x=imaxindex, color='y', linestyle='--')
                ax2.axvline(x=dmaxindex, color='m', linestyle='--')
            f.tight_layout()
        # Other y-axis scale (most likely logarithmic)
        else:
            # Graph
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
            # Plots both infection and death on graph
            ax1.set_title("Total (both)")
            ax1.set_yscale(yscale)
            ax1.set_xlim(0, len(itotal))
            ax1.set_xlabel('days since 1/25')
            itotalline, = ax1.plot(itotal, color='r')
            dtotalline, = ax1.plot(dtotal, color='b')
            ax1.legend((itotalline, dtotalline), ("infection", "death"))
            # Plots both infection and death on graph
            ax2.set_title("Growth (both)")
            ax2.set_yscale(yscale)
            ax2.set_xlim(0, len(idiff))
            ax2.set_xlabel('days since 1/25')
            idiffline, = ax2.plot(idiff, color='r')
            ddiffline, = ax2.plot(ddiff, color='b')
            ax2.legend((idiffline, ddiffline), ("infection", "death"))
            # If graphindex is true, graphs index of maximum daily growth
            if graphindex == True:
                ax1.axvline(x=imaxindex, color='y', linestyle='--')
                ax1.axvline(x=dmaxindex, color='m', linestyle='--')
                ax2.axvline(x=imaxindex, color='y', linestyle='--')
                ax2.axvline(x=dmaxindex, color='m', linestyle='--')
            f.tight_layout()
        print(f"Approximate lag time between infection and death: {lag(iabstot, dabstot, marg)} days")
    # Accounts for bad cases
    else:
        print("Must input one of the following: infection, death, both")
        return


# Compares any two states/provinces/DC/cruise ships

def compareplaces(whichone, p1, p2, percapita=False, k=1, graphindex=False, yscale="Linear"):
    """
    Given place and other parameters, compares two places
        whichone: "infection", "death", "both", determines what array(s) to draw data from
        place: what place to analyze
        percapita: whether data is per capita or in absolute numbers
        k: number of days to take moving average over
        graphindex: whether to graph index of maximum daily growth
        yscale: scale of y-axis (linear, logarithmic)
    """
    # Setting up variables, prints governors' political party
    whichone = whichone.lower()
    yscale = yscale.lower()
    p1results = oneplace(whichone, p1, percapita, k)
    p2results = oneplace(whichone, p2, percapita, k)
    # Graphing one thing: infections or deaths
    if whichone in ["infection", "death"]:
        if k == 1:
            print(f"{p1} vs. {p2}")
            p1total = p1results[1]
            p1differences = p1results[2]
            p2total = p2results[1]
            p2differences = p2results[2]
        else:
            print(f"{p1} vs. {p2} {k}-day average")
            p1total = p1results[2]
            p1differences = p1results[4]
            p2total = p2results[2]
            p2differences = p2results[4]
        print(whatgovernor(p1))
        print(whatgovernor(p2))
        p1maxindex = p1differences.index(max(p1differences))
        p2maxindex = p2differences.index(max(p2differences))
        bothtotal = []
        for i in range(0, len(p1total)):
            bothtotal.append(p1total[i] + p2total[i])
        bothdifferences = []
        for i in range(0, len(p1differences)):
            bothdifferences.append(p1differences[i] + p2differences[i])
        diffbetweentotals = []
        for i in range(0, len(p1total)):
            diffbetweentotals.append(p1total[i] - p2total[i])
        diffbetweendiffs = []
        for i in range(0, len(p1differences)):
            diffbetweendiffs.append(p1differences[i] - p2differences[i])
        # Graph
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7.5))
        # Total
        if whichone == "infection":
            ax1.set_title("Confirmed cases")
        if whichone == "death":
            ax1.set_title("Deaths")
        ax1.set_yscale(yscale)
        ax1.set_xlim(0, len(p1total))
        ax1.set_xlabel('days since 1/25')
        p1totalline, = ax1.plot(p1total, color='r')
        p2totalline, = ax1.plot(p2total, color='b')
        bothtotalline, = ax1.plot(bothtotal, color='g')
        ax1.legend((p1totalline, p2totalline, bothtotalline), (p1, p2, "both"))
        # Growth
        ax2.set_title("Growth")
        ax2.set_yscale(yscale)
        ax2.set_xlim(0, len(p1differences))
        ax2.set_xlabel('days since 1/25')
        p1diffline, = ax2.plot(p1differences, color='r')
        p2diffline, = ax2.plot(p2differences, color='b')
        bothdiffline, = ax2.plot(bothdifferences, color='g')
        ax2.legend((p1diffline, p2diffline, bothdiffline), (p1, p2, "both"))
        # Difference between totals
        ax3.set_title(f"Totals difference ({p1} - {p2})")
        ax3.set_xlim(0, len(p1total))
        ax3.set_xlabel('days since 1/25')
        ax3.plot(diffbetweentotals, color='c')
        ax3.hlines(0, 0, len(p1total), linestyles='dashed')
        ax4.set_title(f"Growth difference ({p1} - {p2})")
        ax4.set_xlim(0, len(p1differences))
        ax4.set_xlabel('days since 1/25')
        ax4.plot(diffbetweendiffs, color='c')
        ax4.hlines(0, 0, len(p1differences), linestyles='dashed')
        if graphindex == True:
            ax1.axvline(x=p1maxindex, color='y', linestyle='--')
            ax1.axvline(x=p2maxindex, color='m', linestyle='--')
            ax2.axvline(x=p1maxindex, color='y', linestyle='--')
            ax2.axvline(x=p2maxindex, color='m', linestyle='--')
        f.tight_layout()
    # Graphing both infections and deaths
    elif whichone == "both":
        if k == 1:
            print(f"{p1} vs. {p2}")
            p1itotal, p1idiff, p1dtotal, p1ddiff, p1iabstot, p1dabstot = p1results[
                1], p1results[2], p1results[4], p1results[5], p1results[6], p1results[7]
            p2itotal, p2idiff, p2dtotal, p2ddiff, p2iabstot, p2dabstot = p2results[
                1], p2results[2], p2results[4], p2results[5], p2results[6], p2results[7]
        else:
            print(f"{p1} vs. {p2} {k}-day average")
            p1itotal, p1idiff, p1dtotal, p1ddiff, p1iabstot, p1dabstot = p1results[
                2], p1results[4], p1results[7], p1results[9], p1results[10], p1results[11]
            p2itotal, p2idiff, p2dtotal, p2ddiff, p2iabstot, p2dabstot = p2results[
                2], p2results[4], p2results[7], p2results[9], p2results[10], p2results[11]
        # Defines necessary variables
        p1imaxindex = p1idiff.index(max(p1idiff))
        p1dmaxindex = p1ddiff.index(max(p1ddiff))
        p2imaxindex = p2idiff.index(max(p2idiff))
        p2dmaxindex = p2ddiff.index(max(p2ddiff))
        bothitotal = []
        diffitotal = []
        for i in range(0, len(p1itotal)):
            bothitotal.append(p1itotal[i] + p2itotal[i])
            diffitotal.append(p1itotal[i] - p2itotal[i])
        bothdtotal = []
        diffdtotal = []
        for i in range(0, len(p1dtotal)):
            bothdtotal.append(p1dtotal[i] + p2dtotal[i])
            diffdtotal.append(p1dtotal[i] - p2dtotal[i])
        bothidiff = []
        diffidiff = []
        for i in range(0, len(p1idiff)):
            bothidiff.append(p1idiff[i] + p2idiff[i])
            diffidiff.append(p1idiff[i] - p2idiff[i])
        bothddiff = []
        diffddiff = []
        for i in range(0, len(p1ddiff)):
            bothddiff.append(p1ddiff[i] + p2ddiff[i])
            diffddiff.append(p1ddiff[i] - p2ddiff[i])
        # Graph
        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(10, 15))
        # Infections
        ax1.set_title("Infections")
        ax1.set_yscale(yscale)
        ax1.set_xlim(0, len(p1itotal))
        ax1.set_xlabel('days since 1/25')
        p1itotalline, = ax1.plot(p1itotal, color='r')
        p2itotalline, = ax1.plot(p2itotal, color='b')
        bothitotalline, = ax1.plot(bothitotal, color='g')
        ax1.legend((p1itotalline, p2itotalline, bothitotalline), (p1, p2, "both"))
        # Deaths
        ax2.set_title("Deaths")
        ax2.set_yscale(yscale)
        ax2.set_xlim(0, len(p1dtotal))
        ax2.set_xlabel('days since 1/25')
        p1dtotalline, = ax2.plot(p1dtotal, color='r')
        p2dtotalline, = ax2.plot(p2dtotal, color='b')
        bothdtotalline, = ax2.plot(bothdtotal, color='g')
        ax2.legend((p1dtotalline, p2dtotalline, bothdtotalline), (p1, p2, "both"))
        # Infection growth
        ax3.set_title("Daily infection growth")
        ax3.set_yscale(yscale)
        ax3.set_xlim(0, len(p1idiff))
        ax3.set_xlabel('days since 1/25')
        p1idiffline, = ax3.plot(p1idiff, color='r')
        p2idiffline, = ax3.plot(p2idiff, color='b')
        bothidiffline, = ax3.plot(bothidiff, color='g')
        ax3.legend((p1idiffline, p2idiffline, bothidiffline), (p1, p2, "both"))
        # Death growth
        ax4.set_title("Daily death growth")
        ax4.set_yscale(yscale)
        ax4.set_xlim(0, len(p1ddiff))
        ax4.set_xlabel('days since 1/25')
        p1ddiffline, = ax4.plot(p1ddiff, color='r')
        p2ddiffline, = ax4.plot(p2ddiff, color='b')
        bothddiffline, = ax4.plot(bothddiff, color='g')
        ax4.legend((p1ddiffline, p2ddiffline, bothddiffline), (p1, p2, "both"))
        # Difference in infection totals
        ax5.set_title(f"Infection totals difference ({p1} - {p2})")
        ax5.set_xlim(0, len(p1itotal))
        ax5.set_xlabel('days since 1/25')
        ax5.plot(diffitotal, color='c')
        ax5.hlines(0, 0, len(p1itotal), linestyles='dashed')
        # Difference in death totals
        ax6.set_title(f"Death totals difference ({p1} - {p2})")
        ax6.set_xlim(0, len(p1dtotal))
        ax6.set_xlabel('days since 1/25')
        ax6.plot(diffdtotal, color='c')
        ax6.hlines(0, 0, len(p1dtotal), linestyles='dashed')
        # Difference in infection daily growths
        ax7.set_title(f"Infection growths difference ({p1} - {p2})")
        ax7.set_xlim(0, len(p1idiff))
        ax7.set_xlabel('days since 1/25')
        ax7.plot(diffidiff, color='c')
        ax7.hlines(0, 0, len(p1idiff), linestyles='dashed')
        # Difference in death daily growths
        ax8.set_title(f"Death growths difference({p1} - {p2})")
        ax8.set_xlim(0, len(p1ddiff))
        ax8.set_xlabel('days since 1/25')
        ax8.plot(diffddiff, color='c')
        ax8.hlines(0, 0, len(p1ddiff), linestyles='dashed')
        # If graphindex is true, graphs index of maximum daily growth
        if graphindex == True:
            ax1.axvline(x=p1imaxindex, color='y', linestyle='--')
            ax1.axvline(x=p2imaxindex, color='m', linestyle='--')
            ax2.axvline(x=p1dmaxindex, color='y', linestyle='--')
            ax2.axvline(x=p2dmaxindex, color='m', linestyle='--')
            ax3.axvline(x=p1imaxindex, color='y', linestyle='--')
            ax3.axvline(x=p2imaxindex, color='m', linestyle='--')
            ax4.axvline(x=p1dmaxindex, color='y', linestyle='--')
            ax4.axvline(x=p2dmaxindex, color='m', linestyle='--')
        f.tight_layout()
        print(f"Approximate lag time between infection and death for {p1}: {lag(p1iabstot, p1dabstot, marg)} days")
        print(f"Approximate lag time between infection and death for {p2}: {lag(p2iabstot, p2dabstot, marg)} days")
    # Accounts for bad cases
    else:
        print("Must input one of the following: infection, death, both")
        return


# Takes total of a list of places

def placestotal(whichone, lst, percapita=False, k=1, graphindex=False, yscale="Linear"):
    """
    Given list of places and other parameters, plots
        whichone: "infection", "death", "both", determines what array(s) to draw data from
        place: what place to analyze
        percapita: whether data is per capita or in absolute numbers
        k: number of days to take moving average over
        graphindex: whether to graph index of maximum daily growth
        yscale: scale of y-axis (linear, logarithmic)
    """
    # Setting up variables, prints governors' political party
    whichone = whichone.lower()
    yscale = yscale.lower()
    pop = 0
    for x in lst:
        pop += population(x)
    # Graphing one thing: infections or deaths
    if whichone in ["infection", "death"]:
        abstotal = [0] * numdays
        for x in lst:
            y = oneplace(whichone, x)
            for i in range(numdays):
                abstotal[i] += y[3][i]
        total = [0] * numdays
        if percapita == True:
            for i in range(numdays):
                total[i] = abstotal[i] / pop
        else:
            total = abstotal
        differences = []
        for i in range(1, numdays):
            differences.append(total[i]-total[i-1])
        if k != 1:
            totalaverage = [0] * numdays
            totalaverage[0] = total[0]
            for i in range(1, k-1):
                totalaverage[i] = mean(total[0:i])
            for i in range(k-1, numdays):
                totalaverage[i] = mean(total[i+1-k:i])
            diffaverage = [0] * numdays
            diffaverage[0] = differences[0]
            for i in range(1, k-1):
                diffaverage[i] = mean(differences[0:i])
            for i in range(k-1, numdays):
                diffaverage[i] = mean(differences[i+1-k:i+1])
            total = totalaverage
            differences = diffaverage
        maxindex = differences.index(max(differences))
        # Graph
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
        # Total
        if whichone == "infection":
            ax1.set_title("Confirmed cases")
        if whichone == "death":
            ax1.set_title("Deaths")
        ax1.set_yscale(yscale)
        ax1.set_xlim(0, len(total))
        ax1.set_xlabel('days since 1/25')
        ax1.plot(total, color='r')
        # Growth
        ax2.set_title("Growth")
        ax2.set_yscale(yscale)
        ax2.set_xlim(0, len(differences))
        ax2.set_xlabel('days since 1/25')
        ax2.plot(differences, color='r')
        if graphindex == True:
            ax1.axvline(x=maxindex, color='y', linestyle='--')
            ax2.axvline(x=maxindex, color='y', linestyle='--')
        f.tight_layout()
    # Graphing both infections and deaths
    elif whichone == "both":
        iabstot = [0] * numdays
        dabstot = [0] * numdays
        for x in lst:
            y = oneplace(whichone, x)
            for i in range(numdays):
                iabstot[i] += y[6][i]
                dabstot[i] += y[7][i]
        iresults = allplaces(iabstot, percapita, k, pop)
        dresults = allplaces(dabstot, percapita, k, pop)
        if k == 1:
            itotal = iresults[0]
            idiff = iresults[1]
            dtotal = dresults[0]
            ddiff = dresults[1]
        else:
            itotal = iresults[1]
            idiff = iresults[3]
            dtotal = dresults[1]
            ddiff = dresults[3]
        imaxindex = idiff.index(max(idiff))
        dmaxindex = ddiff.index(max(ddiff))
        # Defines necessary variables
        if yscale == "linear":
            f, (ax1, ax3) = plt.subplots(1, 2, figsize=(12.8, 4.8))
            # Plots both infection and death on graph but with different y-axes
            ax1.set_title("Total")
            ax1.set_xlabel('days since 1/25')
            ax1.set_xlim(0, len(itotal))
            ax1.set_ylabel('infection', color='r')
            ax1.set_yscale(yscale)
            ax1.tick_params(axis='y', labelcolor='r')
            ax1.plot(itotal, color='r')
            ax2 = ax1.twinx()
            ax2.set_ylabel('death', color='b')
            ax2.set_yscale(yscale)
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.plot(dtotal, color='b')
            # Plots both infection and death growths on graph but with different y-axes
            ax3.set_title("Daily growth")
            ax3.set_xlabel('days since 1/25')
            ax3.set_xlim(0, len(idiff))
            ax3.set_ylabel('infection', color='r')
            ax3.set_yscale(yscale)
            ax3.tick_params(axis='y', labelcolor='r')
            ax3.plot(idiff, color='r')
            ax4 = ax3.twinx()
            ax4.set_ylabel('death', color='b')
            ax4.set_yscale(yscale)
            ax4.tick_params(axis='y', labelcolor='b')
            ax4.plot(ddiff, color='b')
            # If graphindex is true, graphs index of maximum daily growth
            if graphindex == True:
                ax1.axvline(x=imaxindex, color='y', linestyle='--')
                ax1.axvline(x=dmaxindex, color='m', linestyle='--')
                ax3.axvline(x=imaxindex, color='y', linestyle='--')
                ax3.axvline(x=dmaxindex, color='m', linestyle='--')
            f.tight_layout()
        # Other y-axis scale (most likely logarithmic)
        else:
            # Graph
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
            # Plots both infection and death on graph
            ax1.set_title("Total (both)")
            ax1.set_yscale(yscale)
            ax1.set_xlim(0, len(itotal))
            ax1.set_xlabel('days since 1/25')
            itotalline, = ax1.plot(itotal, color='r')
            dtotalline, = ax1.plot(dtotal, color='b')
            ax1.legend((itotalline, dtotalline), ("infection", "death"))
            # Plots both infection and death on graph
            ax2.set_title("Growth (both)")
            ax2.set_yscale(yscale)
            ax2.set_xlim(0, len(idiff))
            ax2.set_xlabel('days since 1/25')
            idiffline, = ax2.plot(idiff, color='r')
            ddiffline, = ax2.plot(ddiff, color='b')
            ax2.legend((idiffline, ddiffline), ("infection", "death"))
            # If graphindex is true, graphs index of maximum daily growth
            if graphindex == True:
                ax1.axvline(x=imaxindex, color='y', linestyle='--')
                ax1.axvline(x=dmaxindex, color='m', linestyle='--')
                ax2.axvline(x=imaxindex, color='y', linestyle='--')
                ax2.axvline(x=dmaxindex, color='m', linestyle='--')
            f.tight_layout()
        print(f"Approximate lag time between infection and death for these places: {lag(iabstot, dabstot, marg)} days")
    # Accounts for bad cases
    else:
        print("Must input one of the following: infection, death, both")
        return


# Compares lists

def comparelists(lst1, lst2, percapita=False, k=1, graphindex=False, yscale="Linear", plotdiff=False):
    """
    Given two lists, compares properties
        lst1: first list
        lst2: second list
        percapita: whether data is per capita or in absolute numbers
        k: number of days to take moving average over
        graphindex: whether to graph index of maximum daily growth
        yscale: scale of y-axis (linear, logarithmic)
    """
    # Setting up variables, title
    title1 = bigtitleall(lst1)
    title2 = bigtitleall(lst2)
    yscale = yscale.lower()
    lst1results = allplaces(lst1, percapita, k)
    lst2results = allplaces(lst2, percapita, k)
    if title1 == title2:
        print(title1)
    else:
        print(f"{title1} (1) vs. {title2} (2)")
    if k == 1:
        total1 = lst1results[0]
        diff1 = lst1results[1]
        total2 = lst2results[0]
        diff2 = lst2results[1]
    else:
        total1 = lst1results[1]
        diff1 = lst1results[3]
        total2 = lst2results[1]
        diff2 = lst2results[3]
    # Defines necessary variables
    diffbetweentotals = []
    for i in range(0, len(total1)):
        diffbetweentotals.append(total1[i] - total2[i])
    diffbetweendiffs = []
    for i in range(0, len(diff1)):
        diffbetweendiffs.append(diff1[i] - diff2[i])
    p1maxindex = diff1.index(max(diff1))
    p2maxindex = diff2.index(max(diff2))
    # Plotdiff is true, i.e., you're comparing two different places
    if plotdiff == True:
        # Graph
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 15))
        # Totals
        ax1.set_title("Totals")
        ax1.set_yscale(yscale)
        ax1.set_xlim(0, len(total1))
        ax1.set_xlabel('days since 1/25')
        lst1line, = ax1.plot(total1, color='r')
        lst2line, = ax1.plot(total2, color='b')
        ax1.legend((lst1line, lst2line), ("1", "2"))
        # Growth
        ax2.set_title("Growth")
        ax2.set_yscale(yscale)
        ax2.set_xlim(0, len(diff1))
        ax2.set_xlabel('days since 1/25')
        diffline1, = ax2.plot(diff1, color='r')
        diffline2, = ax2.plot(diff2, color='b')
        ax2.legend((diffline1, diffline2), ("1", "2"))
        # Difference between totals
        ax3.set_title("Difference between totals")
        ax3.set_xlim(0, len(total1))
        ax3.set_xlabel('days since 1/25')
        ax3.plot(diffbetweentotals, color='c')
        ax3.hlines(0, 0, len(total1), linestyles='dashed')
        # Difference between growths
        ax4.set_title("Difference between growth")
        ax4.set_xlim(0, len(diff1))
        ax4.set_xlabel('days since 1/25')
        ax4.plot(diffbetweendiffs, color='c')
        ax4.hlines(0, 0, len(diff1), linestyles='dashed')
        # If graphindex is true, graphs index of maximum daily growth
        if graphindex == True:
            ax1.axvline(x=p1maxindex, color='y', linestyle='--')
            ax1.axvline(x=p2maxindex, color='m', linestyle='--')
            ax2.axvline(x=p1maxindex, color='y', linestyle='--')
            ax2.axvline(x=p2maxindex, color='m', linestyle='--')
        f.tight_layout()
    # Plotdiff is false, i.e. you're plotting two things from the same place
    else:
        # Linear y-axis scale
        if yscale == "linear":
            # Graph
            f, ((ax1, ax3)) = plt.subplots(1, 2, figsize=(12.8, 4.8))
            # Plots both infection and death on graph but with different y-axes
            ax1.set_title("Total")
            ax1.set_xlabel('days since 1/25')
            ax1.set_xlim(0, len(total1))
            ax1.set_ylabel('1', color='r')
            ax1.set_yscale(yscale)
            ax1.tick_params(axis='y', labelcolor='r')
            ax1.plot(total1, color='r')
            ax2 = ax1.twinx()
            ax2.set_ylabel('2', color='b')
            ax2.set_yscale(yscale)
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.plot(total2, color='b')
            # Plots both infection and death growths on graph but with different y-axes
            ax3.set_title("Daily growth")
            ax3.set_xlabel('days since 1/25')
            ax3.set_xlim(0, len(diff1))
            ax3.set_ylabel('1', color='r')
            ax3.set_yscale(yscale)
            ax3.tick_params(axis='y', labelcolor='r')
            ax3.plot(diff1, color='r')
            ax4 = ax3.twinx()
            ax4.set_ylabel('2', color='b')
            ax4.set_yscale(yscale)
            ax4.tick_params(axis='y', labelcolor='b')
            ax4.plot(total2, color='b')
            # If graphindex is true, graphs index of maximum daily growth
            if graphindex == True:
                ax1.axvline(x=p1maxindex, color='y', linestyle='--')
                ax1.axvline(x=p2maxindex, color='m', linestyle='--')
                ax3.axvline(x=p1maxindex, color='y', linestyle='--')
                ax3.axvline(x=p2maxindex, color='m', linestyle='--')
            f.tight_layout()
        # Other y-axis scale (most likely logarithmic)
        else:
            # Graph
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
            # Totals
            ax1.set_title("Totals")
            ax1.set_yscale(yscale)
            ax1.set_xlim(0, len(total1))
            ax1.set_xlabel('days since 1/25')
            lst1line, = ax1.plot(total1, color='r')
            lst2line, = ax1.plot(total2, color='b')
            ax1.legend((lst1line, lst2line), ("1", "2"))
            # Growths
            ax2.set_title("Growth")
            ax2.set_yscale(yscale)
            ax2.set_xlim(0, len(diff1))
            ax2.set_xlabel('days since 1/25')
            diffline1, = ax2.plot(diff1, color='r')
            diffline2, = ax2.plot(diff2, color='b')
            ax2.legend((diffline1, diffline2), ("1", "2"))
            # If graphindex is true, graphs index of maximum daily growth
            if graphindex == True:
                ax1.axvline(x=p1maxindex, color='y', linestyle='--')
                ax1.axvline(x=p2maxindex, color='m', linestyle='--')
                ax2.axvline(x=p1maxindex, color='y', linestyle='--')
                ax2.axvline(x=p2maxindex, color='m', linestyle='--')
            f.tight_layout()


# Returns sum of a list of places

def allplaces(lst, percapita=False, k=1, pop="placeholder"):
    """
    Returns totals and differences for a list of total cases/deaths
        lst: list
        percapita: whether data is per capita or in absolute numbers
        k: number of days to take moving average over
    """
    if pop == "placeholder":
        pop = population(lst)
    # Defines necessary variables
    if percapita == True:
        total = []
        for i in lst:
            total.append(i/pop)
        lst = total
    differences = []
    for i in range(1, numdays):
        differences.append(lst[i]-lst[i-1])
    # Moving average case
    if k != 1:
        totalaverage = [0] * numdays
        totalaverage[0] = lst[0]
        for i in range(1, k-1):
            totalaverage[i] = mean(lst[0:i])
        for i in range(k-1, numdays):
            totalaverage[i] = mean(lst[i+1-k:i])
        diffaverage = [0] * numdays
        diffaverage[0] = differences[0]
        for i in range(1, k-1):
            diffaverage[i] = mean(differences[0:i])
        for i in range(k-1, numdays):
            diffaverage[i] = mean(differences[i+1-k:i+1])
        return lst, totalaverage, differences, diffaverage
    # Non-average case
    else:
        return lst, differences


# Graphs sum of a list of places

def graphall(lst, percapita=False, k=1, graphindex=False, yscale="Linear"):
    """
    Graphs list of places
        lst: list
        percapita: whether data is per capita or in absolute numbers
        k: number of days to take moving average over
        graphindex: whether to graph index of maximum daily growth
        yscale: scale of y-axis (linear, logarithmic)
    """
    # Prints title and sets up necessary variables
    print(bigtitleall(lst))
    yscale = yscale.lower()
    results = allplaces(lst, percapita, k)
    if k == 1:
        total = lst
        differences = results[1]
    else:
        total = results[1]
        differences = results[3]
    maxindex = differences.index(max(differences))
    # Graph
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    # Total
    if lst in [grandinfectiontotal, sandpinfectiontotal, sanddcinfectiontotal, statesinfectiontotal, gopinfectiontotal]:
        ax1.set_title("Confirmed cases")
    elif lst in [granddeathtotal, sandpdeathtotal, sanddcdeathtotal, statesdeathtotal, gopdeathtotal]:
        ax1.set_title("Deaths")
    ax1.set_yscale(yscale)
    ax1.set_xlim(0, len(total))
    ax1.set_xlabel('days since 1/25')
    ax1.plot(total, color='b')
    # Growth
    ax2.set_title("Growth")
    ax2.set_yscale(yscale)
    ax2.set_xlim(0, len(differences))
    ax2.set_xlabel('days since 1/25')
    ax2.plot(differences, color='g')
    # If graphindex is true, graphs index of maximum daily growth
    if graphindex == True:
        ax1.axvline(x=maxindex, color='r', linestyle='--')
        ax2.axvline(x=maxindex, color='r', linestyle='--')
    f.tight_layout()


# USER-FRIENDLY PART

# Creates a user-friendly interface to analyze data

def userfriendly():
    """
    User-friendly interface to analyze data
    """
    # Asks what to analyze
    whatkind = loop("Do you want to look at\na) one place(can be state, province, DC, or cruise ship),\nb) two places,\nc) the sum of some list of states,\nd) all states with Republican governors,\ne) all States with Democratic governors,\nf) all states (includes option to compare Republican- and Democrat-governed states),\ng) all states and DC,\nh) all states, provinces, and DC, or\ni) all states, provinces, DC, and the cruises?\n(Please input the letter you want.)\n",
                    ["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    # Various cases (see below)
    if whatkind == "A":
        place = getplace(1)
        userfriendlya(place)
    elif whatkind == "B":
        x = getplace(2)
        p1 = x[0]
        p2 = x[1]
        userfriendlyb(p1, p2)
    elif whatkind == "C":
        lst = getplace()
        userfriendlyc(lst)
    else:
        userfriendlydefghi(whatkind)


# Creates loop, continually asking user for input until it is acceptable

def loop(query, words):
    """
    Continually queries user until answer is in given acceptable list
        query: question to ask useer
        words: list of acceptable answers
    """
    x = input(query)
    # Creates infinite loop that can only be broken if the user answers something in the given list
    while True:
        try:
            x = int(x)
        except ValueError:
            x = x.title()
        if x in words:
            break
        else:
            print("Sorry, that input isn't available.")
            x = input(query)
            continue
    return x


# Creates a loop to ask for places until they are ones the program can read

def getplace(n=0):
    """
    Gets either one or two places based on input
    """
    # One place
    if n == 1:
        place = loop("What place would you like to look at?\n", places)
        return place
    # Two places
    elif n == 2:
        p1 = loop("What place would you like to look at first?\n", places)
        p2 = loop("What place would you like to look at second?\n", places)
        while p1 == p2:
            print("Places cannot be the same.")
            p2 = loop("What place would you like to look at second?\n", places)
        return p1, p2
    # List of places
    else:
        lst = []
        place = loop("What place would you like to look at first?\n", places)
        lst.append(place)
        placesandexit = places + ["Exit"]
        while place != "Exit":
            place = loop("What place would you like to look at next (answer 'Exit' to stop)?\n", placesandexit)
            while place in lst:
                print("Places cannot be the same.")
                place = loop("What place would you like to look at next (answer 'Exit' to stop)?\n", placesandexit)
            lst.append(place)
        return lst


# Creates a loop to ask for variables until they are ones the program can read

def helper():
    """
    Helper function that queries the user for several key variables
    """
    # Whether to graph infections, death, or both
    whichone = loop("Would you like to graph infections, death, or both (reply Infection/Death/Both)?\n",
                    ["Infection", "Death", "Both"])
    # Whether to graph absolute or per capita
    pc = loop("Would you like to view the data in absolute numbers or per capita (Absolute/Pc)?\n", ["Absolute", "Pc"])
    percapita = False
    if pc == "Pc":
        percapita = True
    else:
        percapita = False
    # Whether to graph the index of greatest growth
    gi = loop("Would you like to graph the point of greatest growth (Y/N)?\n", ["Y", "N"])
    graphindex = False
    if gi == "Y":
        graphindex = True
    else:
        graphindex = False
    # Whether to graph as a moving average
    avg = loop("Would you like to graph the average across a number of days (Y/N)?\n", ["Y", "N"])
    if avg == "Y":
        k = int(loop("How many days would you like?\n", list(range(numdays))))
    else:
        k = 1
    # What scale on y-axis
    yscale = loop("Would you like to graph on a linear or log scale (Linear/Log)? Note: if a list has no nonzero values, it cannot be graphed on a log scale. American Samoa has no infections or deaths, and the Diamond Princess cruise has no deaths, so if you are plotting either of those, use a linear scale.\n",
                  ["Linear", "Log"])
    return whichone, percapita, graphindex, k, yscale


# Helper function used for one places

def userfriendlya(place):
    """
    Given a place, graphs based on user input
    """
    x = helper()
    whichone, percapita, graphindex, k, yscale = x[0], x[1], x[2], x[3], x[4]
    graphplace(whichone, place, percapita, k, graphindex, yscale)
    return


# Helper function used for two places

def userfriendlyb(p1, p2):
    """
    Compares two given places based on user input
    """
    x = helper()
    whichone, percapita, graphindex, k, yscale = x[0], x[1], x[2], x[3], x[4]
    compareplaces(whichone, p1, p2, percapita, k, graphindex, yscale)
    return


# Helper function used for sum of list of places

def userfriendlyc(lst):
    """
    Compares two given places based on user input
    """
    x = helper()
    whichone, percapita, graphindex, k, yscale = x[0], x[1], x[2], x[3], x[4]
    placestotal(whichone, lst, percapita, k, graphindex, yscale)
    return


# Helper function used for lists of places

def userfriendlydefghi(whatkind):
    """
    Given a group of place, s graphs based on user input
    """
    x = helper()
    whichone, percapita, graphindex, k, yscale = x[0], x[1], x[2], x[3], x[4]
    # Graphing either infection or death
    if whichone in ["Infection", "Death"]:
        if whichone == "Infection":
            if whatkind == "D":
                lst = gopinfectiontotal
            elif whatkind == "E":
                lst = deminfectiontotal
            elif whatkind == "F":
                # Asks if user wants to compare Democrat- and Republican-governed states
                comparepoli = loop("Do you want to compare states governed by Democrats and Republicans (Y/N)?\n", ["Y", "N"])
                if comparepoli == "Y":
                    comparelists(gopinfectiontotal, deminfectiontotal, percapita, k, graphindex, yscale, True)
                    return
                lst = statesinfectiontotal
            elif whatkind == "G":
                lst = sanddcinfectiontotal
            elif whatkind == "H":
                lst = sandpinfectiontotal
            elif whatkind == "I":
                lst = grandinfectiontotal
        elif whichone == "Death":
            if whatkind == "D":
                lst = gopdeathtotal
            if whatkind == "E":
                lst = demdeathtotal
            elif whatkind == "F":
                # Asks if user wants to compare Democrat- and Republican-governed states
                comparepoli = loop("Do you want to compare states governed by Democrats and Republicans (Y/N)?\n", ["Y", "N"])
                if comparepoli == "Y":
                    comparelists(gopdeathtotal, demdeathtotal, percapita, k, graphindex, yscale, True)
                    return
                lst = statesdeathtotal
            elif whatkind == "G":
                lst = sanddcdeathtotal
            elif whatkind == "H":
                lst = sandpdeathtotal
            elif whatkind == "I":
                lst = granddeathtotal
        graphall(lst, percapita, k, graphindex, yscale)
    # Graphs both infections and deaths
    else:
        if whatkind == "D":
            itotal, dtotal = gopinfectiontotal, gopdeathtotal
        if whatkind == "E":
            itotal, dtotal = deminfectiontotal, demdeathtotal
        if whatkind == "F":
            itotal, dtotal = statesinfectiontotal, statesdeathtotal
        if whatkind == "G":
            itotal, dtotal = sanddcinfectiontotal, sanddcdeathtotal
        if whatkind == "H":
            itotal, dtotal = sandpinfectiontotal, sandpdeathtotal
        if whatkind == "I":
            itotal, dtotal = grandinfectiontotal, granddeathtotal
        comparelists(itotal, dtotal, percapita, k, graphindex, yscale)
        print("Infection (1) vs. Death (2)")
        print(f"Approximate lag time between infection and death: {lag(itotal, dtotal, marg)} days")


# RUNS FUNCTION

# Function

userfriendly()
