#pragma once
#include <stdio.h>
#include "dmumps_c.h"

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

extern "C"
{
    CppSolverMumps *new_dmumps()
    {
        printf("...................... new_dmumps\n");
        // return CppSolverMumps::make_new();
        return new CppSolverMumps;
    }

    void drop_dmumps(CppSolverMumps *data)
    {
        printf("###################### drop_dmumps\n");
        delete data;
    }
}
