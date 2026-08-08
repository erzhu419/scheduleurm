************************************************************************
file with basedata            : scheduleurm_mmrcpsp_reserve.bas
initial value random generator: 2
************************************************************************
projects                      :  1
jobs (incl. supersource/sink ):  5
horizon                       :  20
RESOURCES
  - renewable                 :  1   R
  - nonrenewable              :  1   N
  - doubly constrained        :  0   D
************************************************************************
PROJECT INFORMATION:
pronr.  #jobs rel.date duedate tardcost  MPM-Time
    1      3      0       20        0        9
************************************************************************
PRECEDENCE RELATIONS:
jobnr.    #modes  #successors   successors
   1        1          2           2   3
   2        2          1           4
   3        2          1           4
   4        1          1           5
   5        1          0
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration  R 1  N 1
------------------------------------------------------------------------
  1      1     0       0    0
  2      1     5       1    1
         2     2       1    3
  3      1     5       1    1
         2     2       1    3
  4      1     2       2    0
  5      1     0       0    0
************************************************************************
RESOURCEAVAILABILITIES:
  R 1  N 1
    2    4
************************************************************************
