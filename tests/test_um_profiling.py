# Copyright 2025 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import pytest

from access.parsers.um_profiling import UMProfilingParser

@pytest.fixture(scope="module")
def um7_raw_profiling_data():
    """Fixture with raw UM7.x profiling data"""
    return {
        r"""
 FIXED LENGTH HEADER
 -------------------
 Dump format version    20
 UM Version No         703
 Atmospheric data
 Charney-Phillips on radius levels
 Over global domain
*
 23 AP1 Energy Correct.  ****      0.41      0.00      0.41      0.00    1.00
 24 AS18 Assimilation    ****      0.01      0.00      0.02      0.00    0.88
 
 MPP Timing information : 
                   240  processors in configuration                     16  x 
                    15
 
 MPP : Inclusive timer summary
 
 WALLCLOCK  TIMES
    ROUTINE                   MEAN   MEDIAN       SD   % of mean      MAX   (PE)      MIN   (PE)
  1 AS3 Atmos_Phys2        1308.30  1308.30     0.02       0.00%  1308.33 ( 118)  1308.26 ( 221)
  2 AP2 Boundary Layer      956.50   956.14     3.26       0.34%   981.28 ( 136)   953.28 (  43)
  3 AS5-8 Updates           884.63   885.53     2.89       0.33%   889.49 (  48)   879.37 ( 212)
  4 AS2 S-L Advection       746.73   746.73     0.01       0.00%   746.74 (  47)   746.71 ( 181)
  5 AS1 Atmos_Phys1         561.27   562.54    10.63       1.89%   580.32 (  42)   538.58 ( 212)
  6 AP2 Convection          493.73   493.82     0.18       0.04%   493.93 (  76)   493.34 (  20)
 
 CPU TIMES (sorted by wallclock times)
    ROUTINE                   MEAN   MEDIAN       SD   % of mean      MAX   (PE)      MIN   (PE)
  1 AS3 Atmos_Phys2        1308.30  1308.30     0.02       0.00%  1308.33 ( 118)  1308.26 ( 221)
  2 AP2 Boundary Layer      956.50   956.13     3.26       0.34%   981.27 ( 136)   953.28 (  43)
  3 AS5-8 Updates           884.62   885.52     2.89       0.33%   889.49 (  48)   879.36 ( 212)
  4 AS2 S-L Advection       746.72   746.73     0.01       0.00%   746.74 (  47)   746.71 ( 181)
  5 AS1 Atmos_Phys1         561.27   562.53    10.63       1.89%   580.32 (  42)   538.58 ( 212)
  6 AP2 Convection          493.73   493.82     0.18       0.04%   493.93 (  76)   493.34 (  20)
  7 AP1 Radiation           315.24   315.24     0.02       0.01%   315.30 (  66)   315.19 (  13)
  8 AS4 Solver              208.71   208.71     0.01       0.00%   208.72 (  42)   208.69 ( 120)
  9 AS9 End TStep Diags     140.76   140.79     0.12       0.08%   141.00 ( 155)   140.57 (  15)
 10 AP1 Microphysics         65.56    65.56     0.01       0.02%    65.59 (  64)    65.53 (  15)
 11 AS Swap_Bounds           60.83    60.98     1.38       2.27%    62.74 ( 128)    58.21 ( 228)
 12 AP2 River Routing        56.46    56.48     0.09       0.16%    56.66 ( 239)    56.29 (   2)
 13 AEROSOL MODELLING        48.82    48.82     0.02       0.03%    48.87 ( 216)    48.79 ( 109)
 14 DUMPCTL                  13.52    16.88     5.84      43.20%    27.81 (   0)     3.42 (  47)
 15 AP1 G-wave drag           6.17     6.27     0.58       9.45%     7.05 ( 163)     3.71 ( 105)
 16 TIMER                     5.66     5.67     0.13       2.31%     6.02 ( 159)     5.28 ( 115)
 17 AS3 Diffusion             4.64     4.64     0.01       0.11%     4.67 (  15)     4.63 (  16)
 18 AP2 Hydrology             0.50     0.48     0.37      75.19%     1.16 ( 163)     0.03 (  68)
 19 INITDUMP                  0.99     0.99     0.01       0.59%     1.00 ( 198)     0.98 ( 106)
 20 AS9 Energy mass           0.77     0.77     0.00       0.09%     0.78 (   0)     0.77 ( 194)
 21 AP2 Conv Eng Corr         0.58     0.58     0.00       0.84%     0.59 (  15)     0.57 ( 190)
 22 AP1 Conv Eng Corr         0.57     0.57     0.00       0.61%     0.58 (   5)     0.56 ( 135)
 23 AP1 Energy Correct.       0.42     0.42     0.02       3.73%     0.46 (  83)     0.38 ( 215)
 24 AS18 Assimilation         0.01     0.01     0.00       9.45%     0.02 (  64)     0.01 ( 196)
 
 PARALLEL SPEEDUP SUMMARY (sorted by wallclock times)
    ROUTINE              CPU TOTAL   WALLCLOCK MAX   SPEEDUP   PARALLEL EFFICIENCY
  1 AS3 Atmos_Phys2       ********         1308.33    239.99                  1.00
  2 AP2 Boundary Layer    ********          981.28    233.94                  0.97
  3 AS5-8 Updates         ********          889.49    238.69                  0.99
  4 AS2 S-L Advection     ********          746.74    240.00                  1.00
  5 AS1 Atmos_Phys1       ********          580.32    232.12                  0.97
  6 AP2 Convection        ********          493.93    239.90                  1.00
  7 AP1 Radiation         75658.68          315.30    239.96                  1.00
        
        """
    }

@pytest.fixture(scope="module")
def um13_raw_profiling_data():
    """Fixture with raw UM13.x profiling data."""
    return {
        r"""
*******************************************************************************
**************** End of UM RUN Job : 07:34:50 on 27/08/2025 *****************
**************** Based upon UM release vn13.1             *****************
*******************************************************************************


******************************************

END OF RUN - TIMER OUTPUT
Timer information is for whole run
PE 0 Elapsed CPU Time:           1300.190 seconds
PE 0 Elapsed Wallclock Time:     1318.208 seconds
Total Elapsed CPU Time:           758274.502 seconds
*        
MPP Timing information :
576 processors in atmosphere configuration 24 x 24
Number of OMP threads : 1

MPP : Inclusive timer summary

WALLCLOCK  TIMES
N  ROUTINE                                MEAN       MEDIAN        SD   % of mean          MAX  (PE)          MIN  (PE)
01 U_MODEL_4A                          1314.55      1314.55      0.06       0.00%      1315.88 (  0)      1314.55 (433)
02 Atm_Step_4A (AS)                    1272.16      1273.09      4.60       0.36%      1279.04 (240)      1257.69 ( 27)
03 AS Atmos_Phys1 (AP1)                 466.83       466.81      0.21       0.04%       467.36 ( 83)       466.37 (377)
04 AS S-L Advect (AA)                   180.79       181.45      1.67       0.92%       183.17 (104)       175.98 ( 21)
05 AS UKCA_MAIN1                        173.52       174.45      4.60       2.65%       180.40 (240)       159.06 ( 27)
06 AS Atmos_Phys2 (AP2)                 144.45       144.33      2.71       1.87%       150.18 (390)       138.25 (160)

CPU TIMES (sorted by wallclock times)
N  ROUTINE                                MEAN       MEDIAN        SD   % of mean          MAX  (PE)          MIN  (PE)
01 U_MODEL_4A                          1313.66      1314.19      1.65       0.13%      1314.52 (394)      1299.05 (  0)
02 Atm_Step_4A (AS)                    1271.36      1271.79      4.59       0.36%      1278.86 (244)      1249.70 (  0)
03 AS Atmos_Phys1 (AP1)                 466.75       466.79      0.47       0.10%       467.36 ( 83)       462.79 (  0)
04 AS S-L Advect (AA)                   180.78       181.46      1.69       0.93%       183.21 (104)       175.02 (  0)
05 AS UKCA_MAIN1                        173.51       174.45      4.61       2.65%       180.25 (244)       159.06 ( 27)
06 AS Atmos_Phys2 (AP2)                 144.45       144.38      2.70       1.87%       150.25 (390)       138.30 (160)
07 AP1 Radiation (AP1R)                 110.65       110.63      3.30       2.98%       119.59 (296)       103.61 (523)
08 AS Solver                             95.52        95.64      0.90       0.94%        96.80 (  1)        89.48 (339)
09 UKCA AEROSOL MODEL                    70.00        70.01      0.65       0.93%        72.04 (116)        68.18 (574)
10 AP1R LW Rad                           62.11        62.09      3.30       5.32%        71.05 (296)        55.06 (523)
11 AP1 Microphys (AP1M)                  43.50        44.95     11.48      26.38%        68.95 (471)        15.59 (336)
12 COSP                                  65.17        65.31      1.43       2.20%        68.81 ( 90)        60.28 (360)
13 AP2 Convection                        45.68        47.32     11.78      25.80%        67.54 (238)        28.01 (474)
14 AP1M LS Rain                          37.55        38.99     11.47      30.55%        62.99 (471)         9.67 (336)
15 AS CORRECT_TRACERS                    56.52        57.21      1.57       2.78%        58.65 (496)        52.58 ( 14)
16 AA SL_Tracer                          48.02        47.90      0.72       1.50%        50.86 (560)        46.04 (319)
17 AS Stochastic_Phys                    43.20        44.93      4.34      10.05%        46.93 (531)        31.67 (  0)
18 AA SL_Full_Wind                       25.28        24.16      2.15       8.52%        30.94 (555)        23.61 (336)
19 AP2 Boundary Layer                    24.78        24.02      2.39       9.63%        30.11 (390)        19.70 (160)
20 AP1R SW Rad                           23.79        24.28      1.83       7.67%        26.91 (243)        18.23 (563)
21 AP2 Implicit BL                       20.10        19.46      2.37      11.81%        25.12 (390)        14.84 (160)
22 AS End TStep Diags                    23.47        23.36      0.34       1.46%        24.11 (482)        22.93 (279)
23 AA SL_Moisture                        20.66        20.61      0.38       1.82%        21.75 (100)        19.72 (  0)
24 UKCA CHEMISTRY MODEL                  17.97        17.89      0.66       3.65%        21.06 (574)        16.62 ( 29)
25 AS STASH                              17.32        17.31      0.21       1.22%        18.44 ( 90)        16.75 ( 32)
26 INITIAL                               12.21        12.20      0.08       0.69%        12.25 (121)        10.25 (  0)
27 DUMPCTL                                9.30         9.30      0.04       0.43%         9.30 (366)         8.34 (  0)
28 AP1 G-wave drag                        6.49         6.51      0.97      14.92%         9.36 (552)         5.02 (222)
29 TIMER                                  7.70         7.85      0.88      11.46%         9.44 ( 76)         5.94 ( 34)
30 AS CORRECT_MOISTURE                    6.66         6.66      0.02       0.28%         6.75 ( 33)         6.57 (432)
31 AS CONVERT                             0.01         0.00      0.25     ******%         5.98 (  0)         0.00 (  1)
32 AA SL_Rho                              0.01         0.00      0.24     ******%         5.75 (  0)         0.00 (  1)
33 AP2 Explicit BL                        0.01         0.00      0.16     ******%         3.78 (  0)         0.00 (  1)
34 AA SL_Thermo                           0.01         0.00      0.13     ******%         3.23 (  0)         0.00 (  1)
35 AP1M LS Cloud                          0.00         0.00      0.04     ******%         1.01 (  0)         0.00 (  1)
36 AS DIAGNOSTICS                         0.00         0.00      0.04     ******%         1.01 (  0)         0.00 (  1)
37 AS Diffusion                           0.00         0.00      0.02     ******%         0.41 (  0)         0.00 (  1)
38 AS Aerosol Modelling                   0.00         0.00      0.01     ******%         0.29 (  0)         0.00 (  1)
39 AS Assimilation                        0.00         0.00      0.01     ******%         0.19 (  0)         0.00 (  1)
40 AP1 NI_methox                          0.00         0.00      0.01     ******%         0.13 (  0)         0.00 (  1)
41 AS Energy mass                         0.00         0.00      0.01     ******%         0.13 (  0)         0.00 (  1)
42 AP1 Conv Eng Corr                      0.00         0.00      0.00     ******%         0.11 (  0)         0.00 (  1)
43 AP1 Energy Correct.                    0.00         0.00      0.00     ******%         0.10 (  0)         0.00 (  1)
44 AP2 Conv Eng Corr                      0.00         0.00      0.00     ******%         0.10 (  0)         0.00 (  1)
45 AS IAU                                 0.00         0.00      0.00     ******%         0.09 (  0)         0.00 (  1)
46 Init_Atm_Step (FS)                     0.00         0.00      0.00     ******%         0.00 (  0)         0.00 (  1)

?  Caution This run generated 27 warnings
      
        """
    }


@pytest.fixture(scope="module")
def um7_parsed_profile_data():
    """Fixture containing the parsed data with regions, and the associated metrics"""
    return {
        "region": ['AS3 Atmos_Phys2', 'AP2 Boundary Layer', 'AS5-8 Updates', 'AS2 S-L Advection', 'AS1 Atmos_Phys1', 'AP2 Convection'],
        "tavg": [1308.3, 956.5, 884.63, 746.73, 561.27, 493.73],
        "tmed": [1308.3, 956.14, 885.53, 746.73, 562.54, 493.82],
        "tstd": [0.02, 3.26, 2.89, 0.01, 10.63, 0.18],
        "tmax": [1308.33, 981.28, 889.49, 746.74, 580.32, 493.93],
        "pemax": [118, 136, 48, 47, 42, 76],
        "tmin": [1308.26, 953.28, 879.37, 746.71, 538.58, 493.34],
        "pemin": [221, 43, 212, 181, 212, 20],
    }


@pytest.fixture(scope="module")
def um13_parsed_profile_data():
    """Fixture containing the parsed data with regions, and the associated metrics"""
    return {
        "region": ['U_MODEL_4A', 'Atm_Step_4A (AS)', 'AS Atmos_Phys1 (AP1)', 'AS S-L Advect (AA)', 'AS UKCA_MAIN1', 'AS Atmos_Phys2 (AP2)'],
        "tavg": [1314.55, 1272.16, 466.83, 180.79, 173.52, 144.45],
        "tmed": [1314.55, 1273.09, 466.81, 181.45, 174.45, 144.33],
        "tstd": [0.06, 4.6, 0.21, 1.67, 4.6, 2.71],
        "tmax": [1315.88, 1279.04, 467.36, 183.17, 180.4, 150.18],
        "pemax": [0, 240, 83, 104, 240, 390],
        "tmin": [1314.55, 1257.69, 466.37, 175.98, 159.06, 138.25],
        "pemin": [433, 27, 377, 21, 27, 160],
    }


def test_um_metric_names():
    parser = UMProfilingParser()
    assert parser.metrics == ['tavg', 'tmed', 'tstd', 'tmax', 'pemax', 'tmin', 'pemin']

def test_um7_version_parsing(um7_raw_profiling_data):
    parser = UMProfilingParser()
    assert parser.get_um_version(um7_raw_profiling_data) == '7.3', 'Incorrect parsing for UM version'

def test_um13_version_parsing(um13_raw_profiling_data):
    parser = UMProfilingParser()
    assert parser.get_um_version(um13_raw_profiling_data) == '13.1', 'Incorrect parsing for UM version'

def test_um7_parsing(um7_raw_profiling_data, um7_parsed_profile_data):
    parser = UMProfilingParser()
    stats = parser.read(um7_raw_profiling_data)

    # Might also be worthwhile to check that the 'region' key exists first
    assert len(stats['region']) == len(um7_parsed_profile_data['region']), \
        f"Number of matched regions should be *exactly* {len(um7_parsed_profile_data['region'])}"

    metrics = parser.metrics
    for idx, region in enumerate(stats.keys()):
        for metric in metrics:
            assert stats[metric][idx] == um7_parsed_profile_data[metric][idx], \
                f"Incorrect {metric} for region {region} (index: {idx})."

def test_um13_parsing(um13_raw_profiling_data, um13_parsed_profile_data):
    parser = UMProfilingParser()
    stats = parser.read(um13_raw_profiling_data)

    # Might also be worthwhile to check that the 'region' key exists first
    assert len(stats['region']) == len(um13_parsed_profile_data['region']), \
        f"Number of matched regions should be *exactly* {len(um13_parsed_profile_data['region'])}"

    metrics = parser.metrics
    for idx, region in enumerate(stats.keys()):
        for metric in metrics:
            assert stats[metric][idx] == um13_parsed_profile_data[metric][idx], \
                f"Incorrect {metric} for region {region} (index: {idx})."
