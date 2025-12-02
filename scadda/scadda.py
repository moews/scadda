"""Spatio-temporal cluster analysis with density-based distance augmentation.

Introduction:
-------------
SCADDA is a tool for spatio-temporal clustering with density-based distance
re-scaling. Its core is based on the ST-DBSCAN algorithm [1], which is an 
extensions of the common  DBSCAN algorithm [3]. Extensions that are incorporated 
in this software are the re-scaling of the computed distance matrix with kernel 
density estimation over the spatial dimensions and a modulated logistic function, 
as well as time series distance measurements with dynamic time warping, which makes 
use of global constraints via the Sakoe-Chiba band. The algorithm also allows for 
a maximum percentage of outliers to be set, as well as the choice between different 
bandwidths approximations and an option to use vanilla ST-DBSCAN.

As a result, SCADDA is motivated by many real-world spatio-temporal clustering
problems in which the spatial distribution of data points is very varied, with
large numbers of data points in a few centers and sparsely distributed data
points throughout the spatial dimensions. The taken approach allows for taking
these geographical issues into consideration when looking for clusters. This
alleviates an issue with ST-DBSCAN, which considers most data points outside
of such high-density regions as outliers. The method is rooted in prior work
on distance-based rescaling to tackle the problem of spatio-temporal cluster 
analyses for general practicioner data on drug prescription behavior with data 
from National Health Service (NHS) Scotland Open Data [3]. It is a general-purpose 
software tool that can be used to cluster any spatio-temporal dataset with known
latitude and longitude, or with x-axis and y-axis values for the Euclidean
distance, as well as a time series, for each data point.

Quickstart:
-----------
SCADDA is designed to be straightforward to use and needs the following inputs:
(1) The spatial data for each of the data points (name: "s_data")
(2) The time series data for the same data points (name: "t_data")
(3) The distance limit parameter for spatial data (name: "s_limit")
(4) The distance limit parameter for temporal data (name: "t_limit")
(5) The minimum neighbors for a cluster (name: "minimum_neighbors")

A sixth optional parameter is the steepness of the logistic function used to
re-scale the distances (name: "steepness"). This is dependent on the data used,
and the default value is a rule-of-thumb approximation to a sensible steepness.
In addition to these, you can set the window parameter for the width of the
global constraint for the Sakoe-Chiba band (name: "window_param"). If no value
for the parameter is provided, it is set to the level of 10% of the time series
length, as is the common rule of thumb in the related literature [3].

A seventh optional parameter is the distance measure used for the spatial part
of the clustering. As SCADDA was initially developed for longitude and latitude
data, the default choice is the great circle distance. For other applications,
however, the user can set the distance measure (name: "distance_measure") to be
the Euclidean distance, by setting "distance_measure = 'euclidean'".

An eight and final optional parameter is the specification of a maximum
percentage of outliers (name: "outlier_perc"). If set to a number representing
the maximum percentage of outliers present in the results, SCADDA will run
additional iterations over only the the data points classified as outliers in
the first run, identifying additional clusters. Please note that this can be
useful in assigning data points scattered around very high-density peaks, which
would otherwise be outliers, to a separate cluster, but this doesn't mean that
these "second-stage clusters" are clusters in the strict sense of the first
iteration's results. This option should, therefore, be interpreted accordingly.

To start using SCADDA, simply use "from scadda import clutersting" to access
the primary function. An example for using SCADDA looks like this:

    -----------------------------------------------------------------
    |  from scadda import clustering                                |
    |                                                               |
    |  def clustering(s_data = your_spatial_data,                   |
    |                 t_data = your_time_series_data,               |
    |                 s_limit = your_spatial_limit,                 |
    |                 t_limit = your_temporal_limit,                |
    |                 steepness = your_curve_steepness,             |
    |                 minimum_neighbors = your_minimum)             |
    |                                                               |
    -----------------------------------------------------------------

Authors:
--------
Ben Moews & Antonia Gieschen
Business School
The University of Edinburgh

References:
-----------
[1] Birant and Kut (2007), "ST-DBSCAN: An algorithm for clustering spatial-
    temporal data", Data & Knowledge Engineering, 60(1):208-221

[2] Ester et al. (1996), "A density-based algorithm for discovering clusters
   in large spatial databases with noise", KDD-96 Proceedings, 226-231

[3] Gieschen et al. (2022), "Modeling antimicrobial prescriptions in Scotland: 
    A spatiotemporal clustering approach", Risk Analysis, 42(4):830-853

Packages and versions:
----------------------
The versions listed below were used in the development of SCADDA, but the
exact version numbers aren't specifically required. The installation process
via PyPI will take care of installing or updating every library to at least the
level that fulfills the requirement of providing the necessary functionality.

Python 3.4.5
NumPy 1.11.3
SciPy 0.18.1
geopy 1.14.0
pyts 0.13.0
"""
# Import the necessary libraries
import sys
import numpy as numpty
from pyts.metrics import dtw
from scipy.stats import gaussian_kde, zscore
from scipy.spatial.distance import euclidean
from geopy.distance import great_circle

def clustering(s_data,
               t_data,
               s_limit,
               t_limit,
               minimum_neighbors,
               steepness,
               window_param = None,
               distance_measure = None,
               outlier_perc = None,
               z_normalization = None,
               algorithm = None,
               bandwidth = None):
    """
    Assign cluster memberships to a set of user-provided spatio-temporal data.

    This is the primary function of SCADDA, allowing access to its bundled
    functionality with a simple function call. After the spatial and temporal
    distance matrices are computed, the assignments of data points either to
    a cluster or as outliers are calculated and returned to be user.

    Parameters:
    -----------
    s_data : array-like
        Spatial data as latitude and longitude values per data point for the
        default great circle distance, unspecified otherwise. The array should
        have two columns, with the first column as the longitudes / x-axis and
        the second as the latitudes / y-axis, each row being one data point.

    t_data : array-like
        The ime series data as numerical values per data point. The array
        should have as many columns as there are variables, i.e. the number of
        time series of equal length, with each row being one data point.

    s_limit : int or float
        The maximum spatial distance for data points to be considered to be
        part of the same cluster, in kilometers. This applies to the distances
        between data points after the density-based distance re-scaling.

    t_limit : int or float
        The maximum temporal distance for data points to be considered to be
        part of the same cluster, as a dimensionless value. This refers to the
        distance between time series calculated via dynamic time warping.

    minimum_neighbors : int
        The minimum number of neighbors a data point is required to have in
        order to not be considered an outlier. Like the two limit parameters,
        this value depends on the problem to which the code is applied.

    steepness : float
        The steepness of the curve for the logistic function that is used to
        calculate a distance weight as part of the distance re-scaling process.
        A higher value leads to a less extreme alignment of the distances.

    window_param : float, defaults to 0.1 * length(time series)
        The window size for the Sakoe-Chiba band used for putting a global
        constraint on the dynamic time warping. The default value works quite
        well and should only be changed if the other parameters don't work.

    distance_measure : string, defaults to the great circle distance
        The distance measure used for the spatial clustering part of the code.
        By default, SCADDA will use the great circle distance for latidude and
        longitude data, but another option is to use the Euclidean distance.

    outlier_perc : int or float, defaults to 100.0%
        The maximum allowed percentage of outliers for the clustering results.
        If set to, for example, 10.0 for a maximum of 10% of outliers, SCADDA
        will run additional iterations over the remaining outliers to assign
        them to pseudo-clusters. This option should be handled with care. For
        each new iteration, the maximum spatial and temporal distances for data
        points to be in the same cluster is doubled to assure convergence.
        
    z_normalization : boolean, defaults to False
        The indicator whether to apply z-normalisation to the time series 
        associated with each coordinate, meaning the normalization to zero mean
        and unit of energy for comparable means and standard deviations.
    
    algorithm : string, defaults to 'scadda'
        The algorithm to use. This can be either 'scadda', the primary method
        of this package, or 'stdbscan' for the traditional ST-DBSCAN approach 
        that SCADDA is based on, albeit with added dynamic time warping.
        
    bandwidth : string, defaults to 'lscv'

    Returns:
    --------
    assignment : array-like
        Cluster assignments for each data point. The number of clusters is
        determined by the code and doesn't need to be set by the user. It is
        possible that not all clusters will be assigned member data points.

    Attributes:
    -----------
    None
    """
    # If no window parameter is given, set one
    if window_param is None:
        window_param = int(numpty.ceil(len(t_data[0]) / 10))
    # If no distance measure is given, set one
    if distance_measure is None:
        distance_measure = "greatcircle"
    # If no outlier percentage is given, set one
    if outlier_perc is None:
        outlier_perc = 100.0
    # If no normalization is indicated, set to False
    if z_normalization is None:
        z_normalization = False
    # If no algorithm is specified, use SCADDA
    if (algorithm is None):
        algorithm = "scadda"
    # If no bandwidth method is specified, use LSCV
    if (bandwidth is None):
        bandwidth = "scott"
    # Check the input parameters for validity
    parameter_check(s_data = s_data,
                    t_data = t_data,
                    s_limit = s_limit,
                    t_limit = t_limit,
                    minimum_neighbors = minimum_neighbors,
                    steepness = steepness,
                    window_param = window_param,
                    distance_measure = distance_measure,
                    outlier_perc = outlier_perc,
                    z_normalization = z_normalization,
                    algorithm = algorithm,
                    bandwidth = bandwidth)
    # Run a SCADDA iteration to get the cluster assignments
    assignment = iteration(s_data = s_data,
                           t_data = t_data,
                           s_limit = s_limit,
                           t_limit = t_limit,
                           minimum_neighbors = minimum_neighbors,
                           steepness = steepness,
                           window_param = window_param,
                           distance_measure = distance_measure,
                           outlier_perc = outlier_perc,
                           z_normalization = z_normalization,
                           algorithm = algorithm,
                           bandwidth = bandwidth)
    # Provide information on how many outliers were identified in total
    outliers = 100 * ((assignment == -1).sum() / len(assignment))
    print("Percent of data points that are outliers: %.2f\n" % outliers)
    # Shift cluster number up by one to make outliers 0 instead of -1
    assignment = numpty.add(assignment, 1)
    # Shift non-outlier clusters down by one to delete the empty cluster
    for i in range(0, len(assignment)):
        if assignment[i] > 0:
            assignment[i] = assignment[i] - 1
    # If more outliers than allowed, run an additional iteration
    iteration_counter = 1
    if outliers > outlier_perc:
        s_data_copy = s_data.copy()
        t_data_copy = t_data.copy()
    while outliers > outlier_perc:
        print("Too many outliers for the assigned maximum percentage!\n")
        print("Running additional iteration %d ...\n" % iteration_counter)
        # Cut the spatial data to represent only the current outliers
        s_data = s_data_copy[assignment == 0, :]
        # Cut the temporal data to represent only the current outliers
        t_data = t_data_copy[assignment == 0, :]
        # Double allowed distances to allow farther points to be clustered
        s_limit = s_limit * 2
        t_limit = t_limit * 2

        # Re-run the spatio-temporal clustering on the outlier data
        new_assignment = iteration(s_data = s_data,
                                   t_data = t_data,
                                   s_limit = s_limit,
                                   t_limit = t_limit,
                                   minimum_neighbors = minimum_neighbors,
                                   steepness = steepness,
                                   window_param = window_param,
                                   distance_measure = distance_measure,
                                   outlier_perc = outlier_perc,
                                   z_normalizatiion = z_normalization,
                                   algorithm = algorithm)
        # Provide information on how many outliers were identified in total
        outliers = 100 * ((assignment == 0).sum() / len(assignment))
        print("Percent of data points that are outliers: %.2f\n" % outliers)
        # Shift cluster number up by one to make outliers 0 instead of -1
        new_assignment = numpty.add(new_assignment, 1)
        # Get the highest number among the old assignments
        cluster_max = numpty.max(assignment)
        # Shift non-outlier clusters to avoid overlaps with old assignments
        for i in range(0, len(new_assignment)):
            if new_assignment[i] > 0:
                new_assignment[i] = new_assignment[i] + cluster_max - 1
        # Replace old outlier assignments with new cluster assignments
        assignment_counter = 0
        for i in range(0, len(assignment)):
            if assignment[i] == 0:
                if new_assignment[assignment_counter] != 0:
                    assignment[i] = new_assignment[assignment_counter]
                assignment_counter = assignment_counter + 1
        # Update the iteration counter
        iteration_counter = iteration_counter + 1
    print("Complete, now returning cluster assignments!")
    # Return the cluster assignments
    return assignment

def iteration(s_data,
              t_data,
              s_limit,
              t_limit,
              minimum_neighbors,
              steepness,
              window_param,
              distance_measure,
              outlier_perc,
              z_normalization,
              algorithm,
              bandwidth):
    """
    Implement one iteration of computing cluster assignments with SCADDA.

    This is the function incorporating most of SCADDA's specific functionality,
    running one iteration of the algorithm to assign cluster numbers to each
    data point of a given set of spatio-temporal data, respectively.

    Parameters:
    -----------
    s_data : array-like
        Spatial data as latitude and longitude values per data point for the
        default great circle distance, unspecified otherwise. The array should
        have two columns, with the first column as the longitudes / x-axis and
        the second as the latitudes / y-axis, each row being one data point.

    t_data : array-like
        The ime series data as numerical values per data point. The array
        should have as many columns as there are variables, i.e. the number of
        time series of equal length, with each row being one data point.

    s_limit : int or float
        The maximum spatial distance for data points to be considered to be
        part of the same cluster, in kilometers. This applies to the distances
        between data points after the density-based distance re-scaling.

    t_limit : int or float
        The maximum temporal distance for data points to be considered to be
        part of the same cluster, as a dimensionless value. This refers to the
        distance between time series calculated via dynamic time warping.

    minimum_neighbors : int
        The minimum number of neighbors a data point is required to have in
        order to not be considered an outlier. Like the two limit parameters,
        this value depends on the problem to which the code is applied.

    steepness : float
        The steepness of the curve for the logistic function that is used to
        calculate a distance weight as part of the distance re-scaling process.
        A higher value leads to a less extreme alignment of the distances.

    window_param : float
        The window size for the Sakoe-Chiba band used for putting a global
        constraint on the dynamic time warping. The default value works quite
        well and should only be changed if the other parameters don't work.

    distance_measure : string
        The distance measure used for the spatial clustering part of the code.
        By default, SCADDA will use the great circle distance for latidude and
        longitude data, but another option is to use the Euclidean distance.

    outlier_perc : int or float
        The maximum allowed percentage of outliers for the clustering results.
        If set to, for example, 10.0 for a maximum of 10% of outliers, SCADDA
        will run additional iterations over the remaining outliers to assign
        them to pseudo-clusters. This option should be handled with care. For
        each new iteration, the maximum spatial and temporal distances for data
        points to be in the same cluster is doubled to assure convergence.
    
    z_normalization : boolean
        The indicator whether to apply z-normalisation to the time series 
        associated with each coordinate, meaning the normalization to zero mean
        and unit of energy for comparable means and standard deviations.
    
    algorithm : string
        The algorithm to use. This can be either 'scadda', the primary method
        of this package, or 'stdbscan' for the traditional ST-DBSCAN approach 
        that SCADDA is based on, albeit with added dynamic time warping.
        
    bandwidth : string

    Returns:
    --------
    assignment : array-like
        Cluster assignments for each data point. The number of clusters is
        determined by the code and doesn't need to be set by the user. It is
        possible that not all clusters will be assigned member data points.
    """
    # Calculate the spatial distance matrix
    print("Calculating the spatial distance matrix ...\n")
    s_distance = s_similarity_matrix(s_data = s_data,
                                     steepness = steepness,
                                     distance_measure = distance_measure,
                                     algorithm = algorithm,
                                     bandwidth = bandwidth)
    # Apply z-normalization if the user activated it
    if z_normalization == True:
        print("Applying z-normalization to time series ...\n")
        t_data = zscore(t_data, axis = 1)
    # Calculate the temporal distance matrix
    print("Calculating the temporal distance matrix ...\n")
    t_distance = t_similarity_matrix(t_data = t_data,
                                     window_param = window_param)
    cluster, outlier, unassigned, counter = 0, -1, -2, []
    # Initialize cluster assignments as unassigned
    assignment = numpty.full(len(s_distance), unassigned)
    print("Assigning clusters to the provided data ...\n")
    # Loop over the number of provided data points
    for i in range(0, len(s_distance)):
        point = i
        # If unassigned, scan the data point's neighborhood
        if assignment[i] == unassigned:
            neighbors = scan_neighbors(focal_index = i,
                                       s_distance = s_distance,
                                       t_distance = t_distance,
                                       s_limit = s_limit,
                                       t_limit = t_limit)
            # If not enough neighbors, assign outlier status
            if len(neighbors) < minimum_neighbors:
                assignment[i] = outlier
            # If enough neighbors, assign cluster and re-run
            else:
                cluster = cluster + 1
                assignment[i] = cluster
                for neighbor in neighbors:
                    assignment[neighbor] = cluster
                    counter.append(neighbor)
                # Find new neighbors within the center point's neighborhood
                while len(counter) > 0:
                    j = counter[len(counter) - 1]
                    del counter[len(counter) - 1]
                    # Scan the new data point's neighborhood
                    new_neighbors = scan_neighbors(focal_index = j,
                                                   s_distance = s_distance,
                                                   t_distance = t_distance,
                                                   s_limit = s_limit,
                                                   t_limit = t_limit)
                    if len(new_neighbors) > minimum_neighbors - 1:
                        for new_neighbor in new_neighbors:
                            new_cluster = assignment[new_neighbor]
                            # If unassigned, add new cluster
                            if new_cluster == unassigned:
                                assignment[new_neighbor] = cluster
                                counter.append(new_neighbor)
    # Return the assignments
    return assignment
    
def logistic_function(x_value,
                      maximum,
                      midpoint,
                      steepness):
    """
    Calculate a logistic weight for a provided distance between data points.

    This function implements a logistic function with the provided distance
    as the x-axis value, an upper y-axis limit, a midpoint on which the
    function is centered, and a steeness parameter for the function's curve.
    This assignment of y-axis values along a logistic function is used to
    re-scale the distance depending on the midpoint calculated as the mean
    difference of kernel density estimate evaluations of two points over the
    whole dataset and, if the default process for the steepness is used, also
    depending on the number of provided data points.

    Parameters:
    -----------
    x_value : float
        The distance between two data points as the function's x-axis value.

    maximum : float
        The maximum value for the logistic function, i.e. the upper limit.

    midpoint : float
        The x-axis value of the midpoint for the computed logistic function.

    steepness : float
        The steepness of the curve for the logistic function that is used to
        calculate a distance weight as part of the distance re-scaling process.
        A higher value leads to a less extreme alignment of the distances.

    Returns:
    --------
    y_value : float
        The y-axis value for the logistic function generated by the inputs.

    Attributes:
    -----------
    None
    """
    # Calculate the logistic function's output
    y_value = maximum / (1 + numpty.exp(-steepness * (x_value - midpoint)))
    # Return the y-axis value
    return y_value

def t_similarity_matrix(t_data,
                        window_param):
    """
    Calculate the temporal distance matrix for the provided time series data.

    This function calculates the temporal similarity matrix based on dynamic
    time warping with the Sakoe-Chiba band. The computation makes use of the 
    inherent symmetry of the distance matrix in order to only calculate the
    upper triangle and mirror it.

    Parameters:
    -----------
    t_data : array-like
        The ime series data as numerical values per data point. The array
        should have as many columns as there are variables, i.e. the number of
        time series of equal length, with each row being one data point.

    window_param : float
        The window size for the Sakoe-Chiba band used for putting a global
        constraint on the dynamic time warping. The default value works quite
        well and should only be changed if the other parameters don't work.

    Returns:
    --------
    t_distance : array-like
        The final computed distance matrix for the provided time series data.

    Attributes:
    -----------
    None
    """
    # Initialize an empty array to optimize computation
    t_distance = numpty.empty([len(t_data), len(t_data)])
    # Set printout milestones for progress updates
    iter_range = numpty.array_split(numpty.arange(0, len(t_data)), 10)
    print_points = [entry[-1] for entry in iter_range]
    percentage = 0
    # Nested loop over the temporal data to calculate distances
    for i in range(0, len(t_data)):
        for j in range(i + 1, len(t_data)):
            # Calculate a single distance via dynamic time warping 
            t_distance[i, j] = dtw(x = t_data[i, :], 
                                   y = t_data[j, :],
                                   dist='square', 
                                   method='sakoechiba', 
                                   options={'window_size': window_param})
            # Mirror the calculated value to the lower triangle
            t_distance[j, i] = t_distance[i, j]
        # Print progress updates
        if i == print_points[0]:
            percentage = percentage + 10
            print("%d %%" % percentage)
            print_points = print_points[1:len(iter_range)]
    print()
    # Return the temporal distance matrix
    return t_distance

def s_similarity_matrix(s_data,
                        steepness,
                        distance_measure,
                        algorithm,
                        bandwidth):
    """
    Calculate the spatial distance matrix for the provided coordinate data.

    This function calculates the spatial similarity matrix based on grat circle
    distances, which are then re-scaled using kernel density estimation with a
    Gaussian kernel and a bandwidth determined by Scott's rule. Kernel density
    estimate evaluations of the spatial data points are used in combination
    with a modulated logistic function to determine s re-scaling weight for the
    distance between two given points. The code makes use of the symmetry of
    the distance matrix to calculate the upper triangle and mirror it.

    Parameters:
    -----------
    s_data : array-like
        Spatial data as latitude and longitude values per data point for the
        default great circle distance, unspecified otherwise. The array should
        have two columns, with the first column as the longitudes / x-axis and
        the second as the latitudes / y-axis, each row being one data point.

    steepness : float
        The steepness of the curve for the logistic function that is used to
        calculate a distance weight as part of the distance re-scaling process.
        A higher value leads to a less extreme alignment of the distances.

    distance_measure : string
        The distance measure used for the spatial clustering part of the code.
        By default, SCADDA will use the great circle distance for latidude and
        longitude data, but another option is to use the Euclidean distance.
    
    algorithm : string
        The algorithm to use. This can be either 'scadda', the primary method
        of this package, or 'stdbscan' for the traditional ST-DBSCAN approach 
        that SCADDA is based on, albeit with added dynamic time warping.
        
    bandwidth : bandwidth

    Returns:
    --------
    s_distance : array-like
        The final computed distance matrix for the provided coordinate data.

    Attributes:
    -----------
    None
    """
    # Fit a kernel density estimator over the spatial data
    kde = gaussian_kde(dataset = numpty.swapaxes(s_data, 0, 1), 
                       bw_method = bandwidth)
    # Calculate the average density estimate over the spatial data
    kde_mean = numpty.mean(kde(s_data.T))
    # Initialize an empty array to optimize computation
    s_distance = numpty.empty([len(s_data), len(s_data)])
    # Set printout milestones for progress updates
    iter_range = numpty.array_split(numpty.arange(0, len(s_data)), 10)
    print_points = [entry[-1] for entry in iter_range]
    percentage = 0
    # Nested loop over the spatial data to calculate distances
    for i in range(0, len(s_data)):
        for j in range(i + 1, len(s_data)):
            # Evaluate the kernel density estimate for two points
            weight_a = kde(s_data[i, :])
            weight_b = kde(s_data[j, :])
            # Compute the mean of both points' estimates
            weight_mean = numpty.mean([weight_a, weight_b])
            # Calculate a logistic weight for a two-point distance
            logistic_weight = logistic_function(x_value = weight_mean,
                                                maximum = 2,
                                                midpoint = kde_mean,
                                                steepness = steepness)
            # Calculate a single distance with the assigned distance
            if distance_measure == "greatcircle":
                s_distance[i, j] = great_circle(s_data[i, :], s_data[j, :]).km
            elif distance_measure == "euclidean":
                s_distance[i, j] = euclidean(s_data[i, :], s_data[j, :])
            # If using ST-DBSCAN, don't apply logistic weights
            if algorithm == "scadda":
                s_distance[i, j] = s_distance[i, j] * logistic_weight
            # Mirror the calculated value to the lower triangle
            s_distance[j, i] = s_distance[i, j]
        # Print progress updates
        if i == print_points[0]:
            percentage = percentage + 10
            print("%d %%" % percentage)
            print_points = print_points[1:len(iter_range)]
    print()
    # Return the spatial distance matrix
    return s_distance

def scan_neighbors(focal_index,
                   s_distance,
                   t_distance,
                   s_limit,
                   t_limit):
    """
    Find neighbors of a data point to determine whether it's part of a cluster.

    This function scans the neighborhood of a data point for which the index is
    provided, treating the provided data point as the center for the spatial
    and temporal limits provided by the user. If the distances for the data
    points for both spatial and temporal metrics fall within the limites, the
    respective data points are added to the center point's neighborhood.

    Parameters:
    -----------
    focal_index : int
        The index of the point that acts as the center of the search radius.

    s_distance : array-like
        The final computed distance matrix for the provided coordinate data.

    t_distance : array-like
        The final computed distance matrix for the provided time series data.

    s_limit : int or float
        The maximum spatial distance for data points to be considered to be
        part of the same cluster, in kilometers. This applies to the distances
        between data points after the density-based distance re-scaling.

    t_limit : int or float
        The maximum temporal distance for data points to be considered to be
        part of the same cluster, as a dimensionless value. This refers to the
        distance between time series calculated via dynamic time warping.

    Returns:
    --------
    neighbors : array-like
        The neighbors of the given data point depending on the provided limits.

    Attributes:
    -----------
    None
    """
    # Initialize an empty list for neighbors
    neighbors = []
    # Loop over the number of provided data points
    for i in range(0, len(s_distance)):
        scanned_point = i
        if i != focal_index:
            # Retrieve the spatial and temporal distances
            distance_a = s_distance[focal_index, scanned_point]
            distance_b = t_distance[focal_index, scanned_point]
            # If within the distances, add to the neighborhood
            if distance_a < s_limit and distance_b < t_limit:
                neighbors.append(i)
    # Return the data point's neighborhood
    return neighbors

def parameter_check(s_data,
                    t_data,
                    s_limit,
                    t_limit,
                    minimum_neighbors,
                    steepness,
                    window_param,
                    distance_measure,
                    outlier_perc,
                    z_normalization,
                    algorithm,
                    bandwidth):
    """
    Check the user-provided parameter to make sure they are valid inputs.

    This function checks the parameters provided to the primary function to
    avoid any mishaps due to invalid inputs. If one or more parameters don't
    fulfill the requirements of the code, for example due to a wrong format,
    an error notification with an explanation for each invalid input parameter
    is printed, followed by the termination of the function call.

    Parameters:
    -----------
    s_data : array-like
        Spatial data as latitude and longitude values per data point for the
        default great circle distance, unspecified otherwise. The array should
        have two columns, with the first column as the longitudes / x-axis and
        the second as the latitudes / y-axis, each row being one data point.

    t_data : array-like
        The ime series data as numerical values per data point. The array
        should have as many columns as there are variables, i.e. the number of
        time series of equal length, with each row being one data point.

    s_limit : int or float
        The maximum spatial distance for data points to be considered to be
        part of the same cluster, in kilometers. This applies to the distances
        between data points after the density-based distance re-scaling.

    t_limit : int or float
        The maximum temporal distance for data points to be considered to be
        part of the same cluster, as a dimensionless value. This refers to the
        distance between time series calculated via dynamic time warping.

    minimum_neighbors : int
        The minimum number of neighbors a data point is required to have in
        order to not be considered an outlier. Like the two limit parameters,
        this value depends on the problem to which the code is applied.

    steepness : float
        The steepness of the curve for the logistic function that is used to
        calculate a distance weight as part of the distance re-scaling process.
        A higher value leads to a less extreme alignment of the distances.

    window_param : float
        The window size for the Sakoe-Chiba band used for putting a global
        constraint on the dynamic time warping. The default value works quite
        well and should only be changed if the other parameters don't work.

    outlier_perc : float
        The maximum allowed percentage of outliers for the clustering results.
        If set to, for example, 10.0 for a maximum of 10% of outliers, SCADDA
        will run additional iterations over the remaining outliers to assign
        them to pseudo-clusters. This option should be handled with care. For
        each new iteration, the maximum spatial and temporal distances for data
        points to be in the same cluster is doubled to assure convergence.
        
    z_normalization : boolean
        The indicator whether to apply z-normalisation to the time series 
        associated with each coordinate, meaning the normalization to zero mean
        and unit of energy for comparable means and standard deviations.
    
    algorithm : string
        The algorithm to use. This can be either 'scadda', the primary method
        of this package, or 'stdbscan' for the traditional ST-DBSCAN approach 
        that SCADDA is based on, albeit with added dynamic time warping.

    Returns:
    --------
    None

    Attributes:
    -----------
    None
    """
    # Create a vector of boolean values to mark all incorrect inputs
    incorrect_inputs = numpty.zeros(13, dtype = bool)
    # Check if the spatial data is an Nx2 array with float values
    if type(s_data) is not numpty.ndarray:
        incorrect_inputs[0] = True
    elif s_data.shape[1] != 2:
        incorrect_inputs[0] = True
    elif type(s_data[0, 0]) is not numpty.float64:
        incorrect_inputs[0] = True
    # Check if the temporal data is an NxM array with float values
    if type(t_data) is not numpty.ndarray:
        incorrect_inputs[1] = True
    elif type(t_data[0, 0]) is not numpty.float64:
        incorrect_inputs[1] = True
    # Check if spatial and temporal data share the same sample size
    if ((incorrect_inputs[0] is False)
        and (incorrect_inputs[1] is False)):
        if s_data.shape[0] is not t_data.shape[0]:
            incorrect_inputs[2] = True
    # Check if the spatial distance limit is a float or integer > 0
    if ((type(s_limit) is not float)
        and (type(s_limit) is not int)):
        incorrect_inputs[3] = True
    elif s_limit <= 0:
        incorrect_inputs[3] = True
    # Check if the temporal distance limit is a float or integer > 0
    if ((type(t_limit) is not float)
        and (type(t_limit) is not int)):
        incorrect_inputs[4] = True
    elif t_limit <= 0:
        incorrect_inputs[4] = True
    # Check if the number of minimum neighbors is an integer > 0
    if type(minimum_neighbors) is not int:
        incorrect_inputs[5] = True
    elif minimum_neighbors <= 0:
        incorrect_inputs[5] = True
    # Check if the steepness is a float or integer > 0
    if ((type(steepness) is not float)
        and (type(steepness) is not int)):
        incorrect_inputs[6] = True
    elif steepness <= 0:
        incorrect_inputs[6] = True
    # Check if the window parameter is a float or integer > 0
    if ((type(window_param) is not float)
        and (type(window_param) is not int)):
        incorrect_inputs[7] = True
    elif window_param <= 0:
        incorrect_inputs[7] = True
    # Check if the distance measure if among the allowed choices
    if ((type(distance_measure) is not str)
        or distance_measure not in ("greatcircle", "euclidean")):
        incorrect_inputs[8] = True
    # Check if the maximum outlier percentage is a float or integer > 0
    if ((type(outlier_perc) is not float)
        and (type(outlier_perc) is not int)):
        incorrect_inputs[9] = True
    elif outlier_perc <= 0:
        incorrect_inputs[9] = True
    elif outlier_perc > 100:
        incorrect_inputs[9] = True
    # Check if the normalization preference is a valid value
    if type(z_normalization) is not bool:
        incorrect_inputs[10] = True
    # Check if the algorithm specification is a valid input
    if type(algorithm) is not str:
        incorrect_inputs[11] = True
    elif (algorithm != "scadda") and (algorithm != "stdbscan"):
        incorrect_inputs[11] = True
    # Check if the bandwidth estimation method is a valid input
    if ((type(bandwidth) is not float)
        and (type(bandwidth) is not str)):
        incorrect_inputs[12] = True
    elif ((bandwidth != "scott") and (bandwidth != "silverman") 
          and (bandwidth <= 0)):
        incorrect_inputs[12] = True
    # Define error messages for each unsuitable parameter input
    errors = ['ERROR: s_data: Must be an Nx2 numpy.ndarray' +
              'with element type numpy.float64',
              'ERROR: t_data: Must be an NxM numpy.ndarray' +
              'with element type numpyp.float64',
              'ERROR: s_data, t_data: Must have the same number of rows',
              'ERROR: s_limit: Must be a float or integer value > 0',
              'ERROR: t_limit: Must be a float or integer value > 0',
              'ERROR: minimum_neighbors: Must be an integer value > 0',
              'ERROR: steepness: Must be a float or integer value > 0',
              'ERROR: window_param: Must be either None or a' +
              'float or integer value > 0',
              'ERROR: distance_measure: Must be None, or either' +
              ' "greatcircle" or "euclidean"',
              'ERROR: outlier_perc: Must be a float or integer value' +
              ' > 0 and < 100',
              'ERROR: z_normalization: Must be either None or a boolean value',
              'ERROR: algorithm: Must be None or "scadda" or "stdbscan"',
              'ERROR: bandwidth: Must be None, "scott", "silverman" or a float value > 0']
    # If there are unsuitable inputs, print errors and terminate
    if any(value == True for value in incorrect_inputs):
        for i in range(0, len(errors)):
            if incorrect_inputs[i] == True:
                print(errors[i])
        sys.exit()
