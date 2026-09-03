"""
find_material.py

This script queries the Materials Project database to retrieve and compute fully
resolved tensor properties for a selected list of crystalline materials.
It extracts piezoelectric, elastic, and dielectric tensors and carefully processes
them to ensure standard physics conventions (specifically handling Voigt conversions)
before generating a structured JSON output (piezo_data.json) that can be easily parsed.
"""

from mp_api.client import MPRester
from pymatgen.core.tensors import Tensor
import numpy as np
import os
import argparse

# Authentication key for the Materials Project API (v2)
API_KEY = os.environ.get("MAT_PROJ_API_KEY")
if not API_KEY:
    raise ValueError("MAT_PROJ_API_KEY environment variable is not set!")

# Set up argument parsing
parser = argparse.ArgumentParser(description="Fetch and compute tensor properties for Materials Project materials.")
parser.add_argument("materials", nargs="*", default=["661", "2133", "3731", "3666", "6945", "20459", "774922"],
                    help="Material IDs (numbers only, without 'mp-'). E.g., 3731 6945.")
args = parser.parse_args()

with MPRester(API_KEY) as mpr:
    # A curated list of MP material IDs to fetch
    material_ids = [f"mp-{m}" for m in args.materials]
    
    # 1. Fetch Piezoelectric data (Yields the stress-charge tensor 'e')
    piezo_docs = mpr.materials.piezoelectric.search(material_ids=material_ids)
    
    # 2. Fetch Elasticity data (Yields stiffness 'C' and compliance 'S')
    elastic_docs = mpr.materials.elasticity.search(material_ids=material_ids)
    
    # 3. Fetch Dielectric data (Yields permittivity '\epsilon')
    dielectric_docs = mpr.materials.dielectric.search(material_ids=material_ids)
    
    # Hash mapping to easily look up auxiliary properties during the main piezo loop
    elastic_map = {d.material_id: d for d in elastic_docs}
    dielectric_map = {d.material_id: d for d in dielectric_docs}
    
    for doc in piezo_docs:
        print("-" * 60)
        print(f"Material: {doc.material_id} ({doc.formula_pretty})")
        print(f"Symmetry: {doc.symmetry.symbol} (Space Group {doc.symmetry.number})")
        
        # --- PIEZOELECTRIC STRESS TENSOR (e) ---
        # mp-api gives the e-tensor directly in Voigt notation (3x6) in units of C/m^2.
        print("\n--- Piezoelectric Stress Tensor (e) [C/m²] ---")
        print("Voigt Notation (3x6):")
        e_matrix = np.array(doc.total)
        print(e_matrix)
        
        e_doc = elastic_map.get(doc.material_id)
        if e_doc:
            # --- ELASTIC TENSORS (C) ---
            # Stored in the IEEE format directly, which aligns with standard Voigt mappings.
            print("\n--- Elastic Tensors [GPa] ---")
            print("Stiffness Tensor (C):")
            print(np.array(e_doc.elastic_tensor.ieee_format))
        else:
            print("\n--- Elastic Tensors ---")
            print("Elastic data not available for this material.")
            

        d_doc = dielectric_map.get(doc.material_id)
        if d_doc:
            # --- DIELECTRIC PERMITTIVITY TENSOR (\epsilon) ---
            # Stored natively as a 3x3 Cartesian tensor.
            print("\n--- Dielectric Permittivity Tensor (ε) ---")
            print(np.array(d_doc.total))
        else:
            print("\n--- Dielectric Permittivity Tensor ---")
            print("Not available for this material.")

        print("\n")

def export_tensors_to_json(piezo_docs, elastic_map, dielectric_map, filename="piezo_data.json"):
    """
    Consolidates the fetched data and computes the full Cartesian arrays (3x3x3 and 3x3x3x3)
    from their respective Voigt matrices to output a pristine JSON dictionary.
    """
    import json
    from pymatgen.core.elasticity import ElasticTensor, ComplianceTensor
    
    # Establish the root structure with metadata documenting the mapping standard
    data = {
        "_metadata": {
            "voigt_mapping": "11, 22, 33, 23, 13, 12",
            "voigt_mapping_notes": "Standard IEEE mapping: 1->11, 2->22, 3->33, 4->23 (or 32), 5->13 (or 31), 6->12 (or 21).",
            "units": {
                "e_tensor": "C/m^2",
                "e_voigt": "C/m^2",
                "cc_tensor": "GPa",
                "cc_voigt": "GPa",
                "kk_v": "GPa",
                "gg_v": "GPa",
                "kk_r": "GPa",
                "gg_r": "GPa",
                "kk_h": "GPa",
                "gg_h": "GPa",
                "aa_u": "dimensionless",
                "epsilon_tensor": "dimensionless"
            }
        }
    }
    
    for doc in piezo_docs:
        mat_id = doc.material_id
        
        # 1. Expand the Piezoelectric Stress Tensor (e)
        # For the e-tensor, Voigt components map 1:1 with Cartesian components.
        e_voigt = np.array(doc.total)
        e_full = Tensor.from_voigt(e_voigt).tolist()
        
        e_doc = elastic_map.get(mat_id)
        
        # Initialize default values for conditionally available tensors
        c_full = None
        c_voigt_list = None
        kk_v = None
        gg_v = None
        kk_r = None
        gg_r = None
        kk_h = None
        gg_h = None
        aa_u = None
        
        if e_doc:
            # 2. Expand the Elastic Tensor (C)
            # PyMatgen's native classes inherently handle the internal factors of 2 and 4 
            # associated with expanding elastic tensors from Voigt notation.
            c_voigt = np.array(e_doc.elastic_tensor.ieee_format)
            c_full = ElasticTensor.from_voigt(c_voigt).tolist()
            c_voigt_list = c_voigt.tolist()
            

            # Extract Scalar Moduli
            kk_v = float(e_doc.bulk_modulus.voigt) if e_doc.bulk_modulus else None
            gg_v = float(e_doc.shear_modulus.voigt) if e_doc.shear_modulus else None
            kk_r = float(e_doc.bulk_modulus.reuss) if e_doc.bulk_modulus else None
            gg_r = float(e_doc.shear_modulus.reuss) if e_doc.shear_modulus else None
            kk_h = float(e_doc.bulk_modulus.vrh) if e_doc.bulk_modulus else None
            gg_h = float(e_doc.shear_modulus.vrh) if e_doc.shear_modulus else None
            aa_u = float(e_doc.universal_anisotropy) if e_doc.universal_anisotropy is not None else None
            
        # 4. Extract the Dielectric Tensor
        d_doc = dielectric_map.get(mat_id)
        eps_full = np.array(d_doc.total).tolist() if d_doc else None
            
        # 5. Completeness Check
        if any(v is None for v in [e_full, c_full, eps_full, kk_v, gg_v, kk_r, gg_r, kk_h, gg_h, aa_u]):
            print(f"Material {mat_id} ({doc.formula_pretty}) skipped: Not all data available.")
            continue
            
        # Pack the finalized, nested multi-dimensional arrays and metadata into the JSON map
        data[mat_id] = {
            "formula": doc.formula_pretty,
            "crystal_system": str(doc.symmetry.crystal_system),
            "point_group": str(doc.symmetry.point_group),
            "space_group_symbol": str(doc.symmetry.symbol),
            "space_group_number": doc.symmetry.number,
            "e_tensor": e_full,
            "e_voigt": e_voigt.tolist(),
            "cc_tensor": c_full,
            "cc_voigt": c_voigt_list,
            "kk_v": kk_v,
            "gg_v": gg_v,
            "kk_r": kk_r,
            "gg_r": gg_r,
            "kk_h": kk_h,
            "gg_h": gg_h,
            "aa_u": aa_u,
            "epsilon_tensor": eps_full
        }
        
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Successfully exported data with full Cartesian components to {filename}")

# Finally, execute the export sequence
export_tensors_to_json(piezo_docs, elastic_map, dielectric_map)
