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

# Real
# get_matrix Grund b1_ss # Chemical Process Simulation Problem
# get_matrix Bai bfwb62
# get_matrix Bai bfwa398
# get_matrix Bai dwg961a
# get_matrix GHS_psdef inline_1
# get_matrix CEMW tmt_unsym
# get_matrix ATandT pre2
# get_matrix ATandT twotone
# get_matrix Janna Flan_1565
# get_matrix Vavasis av41092
# get_matrix GHS_indef helm2d03
# get_matrix GHS_psdef oilpan
# get_matrix Ronis xenon1
# get_matrix Bai bwm200 # brusselator wave model
# get_matrix Bai bwm2000 # brusselator wave model
# get_matrix Bai rdb5000 # reaction-diffusion brusselator model
# get_matrix Lee fem_filter # complex
# get_matrix Chevron Chevron1
# get_matrix Bai qc324 # complex symmetric
# get_matrix Freescale Freescale1
# get_matrix Freescale memchip
# get_matrix GHS_indef darcy003
# get_matrix GHS_indef boyd2

# Complex
# get_matrix Bai dwg961a # Dispersive waveguide structures
# get_matrix Bai mhd1280b # Alfven spectra in magnetohydrodynamic
# get_matrix Cote mplate # Vibroacoustic problem (plate-air-poroelastic-air-plate system)
# get_matrix Lee fem_filter # FEM bandpass microwave filter, 500MHz
# get_matrix Chevron Chevron3 # Temporal freq domain seismic modeling
# get_matrix Chevron Chevron4 # Temporal freq domain seismic modeling
# get_matrix Kim kim2 # 2D 676-by-676 complex mesh
# get_matrix FreeFieldTechnologies mono_500Hz # 3D vibro-acoustic problem, aircraft engine nacelle
# get_matrix Lee fem_hifreq_circuit # FEM, Maxwell's eqns for hi-freq. circuit
# get_matrix Sinclair 3Dspectralwave # 3D spectral-element elastic wave modelling in freq domain
# get_matrix Dziekonski dielFilterV3clx # High-order vector finite element method in EM