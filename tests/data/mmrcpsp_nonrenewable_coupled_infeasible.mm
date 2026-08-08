************************************************************************
file with basedata            : scheduleurm_mmrcpsp_coupled.bas
initial value random generator: 3
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  5
horizon                       :  20
RESOURCES
  - renewable                 :  1   R
  - nonrenewable              :  2   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1      3      0       20        0        5
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          3           2   3   4
   2        2          1           5
   3        2          1           5
   4        2          1           5
   5        1          0
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  N 1  N 2
------------------------------------------------------------------------
  1      1     0       0    0    0
  2      1     3       1    3    0
         2     3       1    0    3
  3      1     3       1    3    0
         2     3       1    0    3
  4      1     3       1    3    0
         2     3       1    0    3
  5      1     0       0    0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  N 1  N 2
    3    4    4
************************************************************************
