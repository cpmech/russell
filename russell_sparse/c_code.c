#include <stdio.h>
#include <stdlib.h>
#include "dmumps_c.h"

/*
struct DMUMPS_STRUC_C *new_dmumps()
{
    printf("\n##################### new_dmumps\n");
    struct DMUMPS_STRUC_C *data = malloc(sizeof(struct DMUMPS_STRUC_C));
    return data;
}

void drop_dmumps(struct DMUMPS_STRUC_C *data)
{
    printf("\n********************* drop_dmumps\n");
    free(data);
}
*/

struct CppSolverMumps
{
    DMUMPS_STRUC_C data;
    /*
    static CppSolverMumps *make_new()
    {
        return new CppSolverMumps;
    }
    */
};

struct CppSolverMumps *new_dmumps()
{
    printf("...................... new_dmumps\n");
    struct CppSolverMumps *data = (struct CppSolverMumps *)malloc(sizeof(struct CppSolverMumps));
    return data;
    // return CppSolverMumps::make_new();
    // return new CppSolverMumps;
}

void drop_dmumps(struct CppSolverMumps *data)
{
    printf("###################### drop_dmumps\n");
    // delete data;
    free(data);
}
