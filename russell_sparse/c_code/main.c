#include <stdio.h>
#include <stdlib.h>
#include "dmumps_c.h"

struct CppSolverMumps
{
    DMUMPS_STRUC_C data;
};

struct CppSolverMumps *new_dmumps()
{
    struct CppSolverMumps *data = (struct CppSolverMumps *)malloc(sizeof(struct CppSolverMumps));
    return data;
}

void drop_dmumps(struct CppSolverMumps *data)
{
    free(data);
}
