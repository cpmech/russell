#!/bin/bash

# see: https://sparse.tamu.edu/

set -e

cd ~/Downloads
mkdir -p matrix-market
cd matrix-market

get_matrix() {
  GROUP=$1
  NAME=$2
  wget https://suitesparse-collection-website.herokuapp.com/MM/$GROUP/$NAME.tar.gz
  tar xzf $NAME.tar.gz
  mv $NAME/$NAME.mtx .
  rm -rf $NAME
  rm $NAME.tar.gz
}

# Note: pres-cylin-3d-tet10-fine is NOT from the SuiteSparse Collection.
# It must be obtained separately (see README.md).

# Real matrices
get_matrix Bai bwm2000             # Brusselator wave model in transport interaction of chemical solutions
get_matrix Bai rdb5000             # Reaction-diffusion brusselator model
get_matrix Goodwin Goodwin_040     # Finite element, Navier-Stokes & other transport equations
get_matrix MKS fp                  # 2-D Fokker Planck eqn, electron dyn. in external field
get_matrix Ronis xenon1            # Complex zeolite, sodalite crystals
get_matrix ATandT twotone          # Harmonic balance method
get_matrix Rajat Raj1              # Circuit Simulation Problem
get_matrix GHS_indef boyd2         # Optimization problem
get_matrix Goodwin Goodwin_071     # Finite element, Navier-Stokes & other transport equations
get_matrix GHS_indef darcy003      # Discretization using mixed FE of Darcy
get_matrix Bova rma10              # 3D CFD model, Charleston harbor
get_matrix GHS_indef helm2d03      # Helmholtz eq on a unit square
get_matrix Norris stomach          # 3D electro-physical model of a duodenum
get_matrix GHS_psdef oilpan        # Structural problem
get_matrix Sandia ASIC_680k        # Circuit simulation matrix
get_matrix CEMW tmt_unsym          # Electromagnetics problem
get_matrix Goodwin Goodwin_127     # Finite element, Navier-Stokes & other transport equations
get_matrix ATandT pre2             # Harmonic balance method
get_matrix Martin marine1          # Chemical oceanography; a marine nitrogen cycle inverse model
get_matrix Norris torso1           # Finite differences and boundary element, 2D model of torso
get_matrix Bourchtein atmosmodd    # CFD analysis of atmospheric models
get_matrix Bourchtein atmosmodl    # CFD analysis of atmospheric models
get_matrix Freescale memchip       # Circuit simulation problem
get_matrix Freescale Freescale1    # Circuit simulation problem
get_matrix Rajat rajat31           # Circuit simulation problem
get_matrix Janna Transport         # 3D finite element flow and transport
get_matrix GHS_psdef inline_1      # Structural problem, stiffness matrix
get_matrix Janna PFlow_742         # 3D pressure-temperature evolution in porous media
get_matrix Janna Emilia_923        # Geomechanical model for C02 sequestration
get_matrix Dziekonski dielFilterV2real # FEM in electromagnetics
get_matrix Janna Flan_1565         # Structural problem, 3D model of a steel flange

# Complex matrices
get_matrix Bai mhd1280b                    # Alfven spectra in magnetohydrodynamic
get_matrix Cote mplate                     # Vibroacoustic problem (plate-air-poroelastic-air-plate system)
get_matrix Rost RFdevice                   # Semiconductor device simulation
get_matrix CEMW vfem                       # Electromagnetics Vector finite element
get_matrix Lee fem_filter                  # FEM bandpass microwave filter, 500MHz
get_matrix Chevron Chevron4                # Temporal freq domain seismic modeling
get_matrix FreeFieldTechnologies mono_500Hz # 3D vibro-acoustic problem, aircraft engine nacelle
get_matrix Kim kim2                        # 2D 676-by-676 complex mesh
get_matrix Lee fem_hifreq_circuit          # FEM, Maxwell's eqns for hi-freq. circuit
get_matrix Dziekonski dielFilterV3clx      # High-order vector finite element method in EM
