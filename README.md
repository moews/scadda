# SCADDA

### Spatio-temporal cluster analysis with density-based distance augmentation

<img src="/logo.png" alt="logo" width="200px"/>

SCADDA is a tool for spatio-temporal clustering with density-based distance re-scaling. Its core, an extended implementation of methodology introduced by [Gieschen et al. (2022)](https://doi.org/10.1111/risa.13795), is based on [ST-DBSCAN](https://dl.acm.org/citation.cfm?id=1219397), which is a previous modification of the common [DBSCAN](https://dl.acm.org/citation.cfm?id=3001507) algorithm. Extensions that are incorporated in this new method are the re-scaling of the computed distance matrix with kernel density estimation over the spatial dimensions and a modulated logistic function, as well as time series distance measurements with dynamic time warping (DTW), which makes use of global constraints via the [Sakoe-Chiba band](https://ieeexplore.ieee.org/document/1163055/). In addition, SCADDA also features the functionality to be used as an implementation of the unmodified traditional [ST-DBSCAN](https://dl.acm.org/citation.cfm?id=1219397) method.

As a result, SCADDA is motivated by many real-world spatio-temporal clustering problems in which the spatial distribution of data points is very varied, with large numbers of data points in a few centers and sparsely distributed data points throughout the spatial dimensions. This approach allows for taking these geographical issues into consideration when looking for clusters, alleviating the issue that [ST-DBSCAN](https://dl.acm.org/citation.cfm?id=1219397) considers most data points outside of such high-density regions as outliers. It is a general-purpose software tool that can be used to cluster any spatio-temporal dataset with known latitude and longitude, as well as a time series for a variable, for each data point. 

<!--
### Installation

SCADDA can be installed via [PyPI](https://pypi.org), with a single command in the terminal:

```
pip install scadda
```

Alternatively, the file `scadda.py` can be downloaded from the folder `scadda` in this repository and used locally by placing the file into the working directory for a given project. An installation via the terminal is, however, highly recommended, as the installation process will check for the package requirements and automatically update or install any missing dependencies, thus sparing the user the effort of troubleshooting and installing them themselves.
-->

### Quickstart guide

SCADDA requires the user to provide spatial data (`s_data`) as a Nx2 array for N data points, with longitudes in the first and latitudes in the second column, as well as the same number of time series per spatial data point (`t_data`) as an NxM array with M as the length of the time series. 

The spatial (`s_limit`) and temporal (`t_limit`) maximal distances for points to be considered part of the same cluster, as well as the steepness for the logistic function used for the distance re-scaling (`steepness`) and the mininum number of neighbors required for a cluster (`minimum_neighbors`), also have to be provided. 

In addition, the window size for the [Paliwal adjustment window](https://ieeexplore.ieee.org/document/1171506/) can be set (`window_param`) by the user, but this parameter is optional and will default to a data-dependent rule-of-thumb calculation.

Lastly, four additional optional parameters can be set: The distance measure (`distance_measure`) uses the great circle distance for longitudes and latitudes is used by default, but can be set to either "greatcircle" or "euclidean". 

If a maximum percentage of outliers (`outlier_perc`) is provided, SCADDA will run additional iterations over the remaining outliers to assign them to pseudo-clusters. For each new iteration, the maximum spatial and temporal distances for data points to be in the same cluster are doubled to assure convergence.

An additional boolean parameter can be set to apply *z*-score normalization to the provided time series (`z_normalization`) to enforce that all time series share a common mean of approximately zero and a common standard deviation of approximately one. The last optional parameter (`algorithm`), with the possible inputs "scadda" and "stdbscan", can be used to employ either the primary method of this package or the traditional ST-DBSCAN algorithm it extends, with the same DTW-based distance measure for time series. Parameters are listed below.

<br></br>

| Variables                    | Explanations                                        | Default               |
|:-----------------------------|:----------------------------------------------------|:----------------------|
| s_data                       | The spatial data, i.e. the data point coordinates   |                       |
| t_data                       | The temporal data, i.e. time series per data point  |                       |
| s_limit                      | The maximum same-cluster spatial distance           |                       |
| t_limit                      | The maximum same-cluster temporal distance          |                       |
| steepness                    | The curve steepness density-based distance weights  |                       |
| minimum_neighbors            | The minimum number of neighbors for non-outliers    |                       |
| window_param (optional)      | The window size for the Sakoe-Chiba band for DTW    | 0.1 * length(t_data)  |
| distance_measure (optional)  | The distance measure for the spatial clustering     | "greatcircle"         |
| outlier_perc (optional)      | The maximum allowed percentage of outliers          | 100                   |
| z_normalization (optional)   | The choice to apply z-normalization to time series  | False                 |
| algorithm (optional)         | The algorithm to use for clustering the dataset     | "scadda"              |

<br></br>

<!--
After the installation via [PyPI](https://pypi.org), or using the `scadda.py` file locally, the usage looks like this:
-->

### Use case example

```python
from scadda import clustering

cluster_assignments = clustering(s_data = your_spatial_data,
                                 t_data = your_time_series_data,
                                 s_limit = your_spatial_limit,
                                 t_limit = your_temporal_limit,
                                 steepness = your_curve_steepness,
                                 minimum_neighbors = your_minimum)
```
