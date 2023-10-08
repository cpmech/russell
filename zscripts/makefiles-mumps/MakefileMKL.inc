#
#  This file is part of MUMPS 5.6.1, released
#  on Tue Jul 11 07:51:28 UTC 2023
#
# These settings for a PC under Debian/linux with standard packages :
# metis (parmetis), scotch (ptscotch), openmpi 

# must be at the top
PLAT = _cpmech

# Begin orderings
LSCOTCHDIR = 
ISCOTCH   = 

LSCOTCH   = 

LPORDDIR = $(topdir)/PORD/lib/
IPORD    = -I$(topdir)/PORD/include/
LPORD    = -L$(LPORDDIR) -lpord$(PLAT)

LMETISDIR = /usr/lib 
IMETIS    = -I/usr/include/metis

LMETIS    = -L$(LMETISDIR) -lmetis

# Corresponding variables reused later
ORDERINGSF = -Dmetis -Dpord
ORDERINGSC  = $(ORDERINGSF)

LORDERINGS = $(LMETIS) $(LPORD) $(LSCOTCH)
IORDERINGSF = $(ISCOTCH)
IORDERINGSC = $(IMETIS) $(IPORD) $(ISCOTCH)
# End orderings
################################################################################

LIBEXT_SHARED  = .so
SONAME = -soname
FPIC_OPT = -fPIC
# Adapt/uncomment RPATH_OPT to avoid modifying
# LD_LIBRARY_PATH in case of shared libraries
# RPATH_OPT = -Wl,-rpath,/path/to/MUMPS_x.y.z/lib/
LIBEXT  = .a
OUTC    = -o 
OUTF    = -o 
RM = /bin/rm -f
CC = gcc
FC = ifort
FL = ifort
AR = ar vr 
RANLIB = ranlib
MKLROOT=/opt/intel/oneapi/mkl/latest/lib/intel64
LAPACK = -L$(MKLROOT) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core



INCSEQ = -I$(topdir)/libseq
LIBSEQ  = $(LAPACK) -L$(topdir)/libseq -lmpiseq$(PLAT)

LIBBLAS = -L$(MKLROOT) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core 
LIBOTHERS = -lpthread

#Preprocessor defs for calling Fortran from C (-DAdd_ or -DAdd__ or -DUPPER)
CDEFS   = -DAdd_

#Begin Optimized options
OPTF    = -O -nofor-main -fopenmp -DGEMMT_AVAILABLE
OPTL    = -O -nofor-main -fopenmp
OPTC    = -O -fopenmp
#End Optimized options
 
INCS = $(INCSEQ)
LIBS = $(LIBSEQ)
LIBSEQNEEDED = libseqneeded
