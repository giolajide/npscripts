"""
For strain and CN analyses\n
Writes the output CSV that will be plotted later\n
Also writes the traj file with the system_type alone there


This script tracks heights by what layer each atom belongs to

    In this version, for height-varying strain, we consider bonds
    from atom A to only its neighbors that are in the same layer as A

September 4th: I also implement it calculating volumetric strains
                i.e. also considering interkayer bonds



NOTE:
    wherever, in this document, I just say 'strain',
    I mean 'volumetric strain'
"""
from argparse import ArgumentParser
from copy import deepcopy
from ase.io import read, write
import pandas as pd
from numba import prange, jit
import numpy as np
from ase import Atoms
from tqdm import tqdm
from typing import List, Tuple, Union
#from ase.visualize import view
import warnings
from ase.neighborlist import NeighborList, natural_cutoffs
from matscipy.neighbours import neighbour_list
from utilities.sphere_and_scm_conversions import _natoms_to_SCM_footprint_diameter
from utilities.print_bonds import setup_neighborlist#, setup_analyzer
from catalytic_descriptors import (
        FCC_AND_HCP_COORD_CUTOFF, #12
        get_np_surface_by_CN,
        get_interface, TAGS_DICT
        )
from fit_support import divider
from utilities.utilities import classify_into_layers, LAYER_TOLERANCE

THETA = 106 #default contact angle
DEFAULT_MLIP_AG_LATTICE = 4.102739436893522 #by MLIP
GPAW_AG_LATTICE = 4.1040929982444405 #in case I need it sometime
MULT = 1.
EXTRA_TAGS = {
        "all": [0, 1, 2, 3], #excludes the substrate, of course!
        "surface_perimeter_interface": [1, 3, 2],
        "perimeter_interface": [3, 2],
        }
TAGS_DICT.update(EXTRA_TAGS)
EXPANDED_TAGS_DICT = deepcopy(TAGS_DICT)


######################################### COORDINATION NUMBERS ###########################################
def get_CN(
        atoms: Atoms,
        mult: float = 1,
        radial_cutoff: float = None
        ):
    """
    Get coordination numbers, connectivity matrix,
    and neighborlist object using ASE's neighborlist
    implementation

    Requires:
        atoms:      Atoms object
        mult:       scaling factor for covalent radii

    if radial_cutoff is given, then we will consider only neighbors within
    the lower of that cutoff and the covalent/ionic bond length

    Returns:
        CN:          array of coordination number of each atom
        CN_matrix:  connectivity matrix, i.e. symmetric matrix of 1s or 0s
                    telling if atom i is connected to atom j
        nl:         neighborlist object
    """
    atoms, _ = divider(atoms)
    nl = setup_neighborlist(atoms = atoms,
            scaling_factor = mult,
            radial_cutoff = radial_cutoff
            )

    CN_matrix = nl.get_connectivity_matrix(sparse=False)
    CN = CN_matrix.sum(axis=0)

    return CN, CN_matrix, nl


def get_CN_fast(
        atoms: Atoms,
        mult: float = 1,
        cutoffs: Union[np.ndarray, float] = None
        ) -> np.ndarray:
    """
    Get coordination numbers, connectivity matrix,
    and neighborlist object using matscipy's neighborlist
    implementation

    Requires:
        atoms:      Atoms object
        mult:       scaling factor for covalent radii

    if radial_cutoff is given, then we will consider only neighbors within
    the lower of that cutoff and the covalent/ionic bond length

    Returns:
        CN:          array of coordination number of each atom
    """
    atoms, _ = divider(atoms)
    warnings.warn("This function seems to give wrong answers",
            category = UserWarning)

    if cutoffs is None:
        cutoffs = natural_cutoffs(atoms)

    cutoffs = np.array(cutoffs)
    cutoffs *= mult

    i, j = neighbour_list("ij", atoms, cutoffs)
    CN = np.bincount(i, minlength=len(atoms))

    return CN


@jit(parallel = True, nopython = True)
def get_GCN(
        CN: np.ndarray,
        CN_matrix: np.ndarray
        ) -> np.ndarray: #, cn_max: float=None):
    """
    Get generalized CN
    GCN(i) = SUM(CN(j) / CNmax)
    i.e. the GCN of atom i is the average CN of its neighbors compared to their 
    maximum possible CN

    Requires:
        CN:          array of coordination number of each atom
        CN_matrix:  connectivity matrix, i.e. symmetric matrix of 1s or 0s
                    telling if atom i is connected to atom j
    Returns:
        GCN:        array of generalized coordination number of each atom
    """
    if True: #cn_max is None:
        cn_max = CN_matrix.sum(axis=0).max()
    if cn_max != FCC_AND_HCP_COORD_CUTOFF:
        print(
        f"""Max CN ({cn_max}) != {FCC_AND_HCP_COORD_CUTOFF}!\n
        This might be a very smll cluster\n
        OR\n
        Something might be very wrong here!"""
        )

    num_atoms = CN_matrix.shape[0]
    GCN = np.zeros(num_atoms)
    #multithread the outer loop not the inner one or there might
    #be a race condition
    for A in prange(num_atoms):
        for B in range(num_atoms):
            GCN[A] += CN_matrix[A, B] * CN[B]
        GCN[A] /= cn_max

    return GCN


############################################## STRAIN #############################################
def compute_strain(
        atoms: Atoms,
        *, #must specify by keyword every arg thence
        nl: NeighborList,
        system_type: str,
        CN: np.ndarray = None,
        lattice_constant: float = DEFAULT_MLIP_AG_LATTICE,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f"""
    Calculate strains, surface strains, and their dependence on height

    Requires:
        atoms:              ENTIRE Nanoparticle, i.e not outer surface only
        nl:                 ASE neighborlist object
        lattice_constant:   lattice constant.
                            default: {DEFAULT_MLIP_AG_LATTICE}
        system_type:        type of system analyzed, MUST be either:
                            'all', 'bulk', 'interface', 'perimeter', 'surface', 'substrate',
                            'surface_perimeter_interface', or
                            'perimeter_interface'
                            Makes no sense to request 'substrate', btw
        CN:                 array of coordination numbers

    Returns:
        strains:            array of % volumetric strains
        surface_strains:    array of % volumetric surface strains
        normal_strains:     array of normal strains. see top of script for definition
        layer_percents:     array of height %
    """
    atoms, _ = divider(atoms) #in case
    ##get the layers
    atoms, layers = classify_into_layers(atoms, tol = LAYER_TOLERANCE)
    ##normalize layers
    max_layers = np.max(layers)
    layer_percents_all = 100 * layers/max_layers #from 0 to 100%;
                                                #interface will be from 0% to (100 * 1/max(layers))%
    layer_percents = []
    system_type = system_type.lower()
    requested_tag = EXPANDED_TAGS_DICT[system_type]
    if not isinstance(requested_tag, list):
        requested_tag = [requested_tag]

    lattice_constant = lattice_constant if lattice_constant\
            else DEFAULT_MLIP_AG_LATTICE

    if (CN is None or nl is None):
        CN, _, nl = get_CN(atoms)

    d_ideal = lattice_constant / np.sqrt(2) ##assumes FCC
    surface_strains, strains = [], []
    ##this is the normal rather than volumetric strains
    ##it's the best for comparison with TEM images
    normal_strains = []

    bulk_indices = [
            i for i, j in enumerate(CN)
            if j >= FCC_AND_HCP_COORD_CUTOFF
            ]
   
    all_to_show = []

    for index, atom in enumerate(atoms):
        to_show = []
        if atoms.get_tags()[index] in requested_tag:

            current_layer = layers[index]
            layer_percents.append(layer_percents_all[index])

            #calculate strain
            neighbors = nl.get_neighbors(index)[0]
            distances = atoms.get_distances(index,
                    indices=neighbors,
                    mic=True
                    )
            atom_strains = (distances - d_ideal) / d_ideal #negative if compressive
            strains.append(np.nanmean(atom_strains) * 100)


            #calculate normal strain
            ##no point in calculating normal surface strain, as TEM sees not that
            #extract just the neighbors in the same layer
            samelayer_indices = np.where(layers == current_layer)[0]
            samelayer_neighbors = np.intersect1d(neighbors, samelayer_indices)

            ##just in case there is no same-layer neighbor
            if len(samelayer_neighbors) == 0:
                warnings.warn(f"Atom {index} has no same-layer neighbors!\nSetting its normal strain to NaN",
                        category = UserWarning
                        )
                normal_strains.append(np.nan)
            else:
                samelayer_distances = atoms.get_distances(index,
                        indices=samelayer_neighbors,
                        mic=True
                        )
                atom_normal_strains = (samelayer_distances - d_ideal) / d_ideal #negative if compressive
                normal_strains.append(np.nanmean(atom_normal_strains) * 100)


            #calculate surface strain
            neighbors = [
                    neighbor for neighbor in neighbors
                    if neighbor not in bulk_indices
                    ]
        
            if len(neighbors) == 0:
                warnings.warn(f"Atom {index} has no surface neighbors!\nSetting its surface strain to NaN",
                        category = UserWarning
                        )
                surface_strains.append(np.nan)
                to_show.append(atom)
            else:
                distances_surf = atoms.get_distances(
                        index,
                        indices=neighbors,
                        mic=True
                        )

                atom_surf_strains = (distances_surf - d_ideal) / d_ideal #negative if compressive
                surface_strains.append(np.nanmean(atom_surf_strains) * 100)

        all_to_show.append(Atoms(to_show))
    #    view(all_to_show)

    return np.array(strains), np.array(surface_strains), np.array(normal_strains), np.array(layer_percents)



############################## OUTPUT FILE ################################################################
def write_output(
        traj: Union[List[Atoms], Atoms],
        system_: str = None,
        system_type: str = None,
        mult: float = 1,
        a: float = DEFAULT_MLIP_AG_LATTICE,
        radial_cutoff: float = None,
        theta: float = THETA,
        ) -> List[Atoms]:
    f"""
    Writes the output CSV that will be plotted later
    Also writes the traj file with the system_type alone there

    Requires:
        traj:               Trajectory file of the ENTIRE Nanoparticle,
                            i.e not outer surface only
        system_:            Name of output file prefix
                            I would normally expect 'Optimized', 'Wulff',
                            'Winter', or 'Plato'
        system_type:        type of system analyzed, MUST be either:
                            'bulk', 'interface', 'perimeter', 'surface', 'substrate',
                            'surface_perimeter_interface', or
                            'perimeter_interface'
                            Makes no sense to request 'substrate', btw
        mult:               scaling factor for covalent radii
        a:                  lattice constant. default: {DEFAULT_MLIP_AG_LATTICE}
        theta:              contact angle in degrees. default: {THETA}

    if radial_cutoff is given, then we will consider only neighbors within
    the lower of that cutoff and the covalent/ionic bond length

    Returns:
        output_traj:        the traj file with the system_type alone there

    Does:
        Writes the output CSV that will be plotted later
        Also writes the traj file with the system_type alone there
    """
    if isinstance(traj, Atoms):
        traj = [traj]
    system_ = system_.capitalize()

    n_atoms, r_atoms, diameter = [], [], []
    CNs = {i: [] for i in range(1, 13)}
    GCNs = {i: [] for i in range(1, 13)}
    SurfToBulks = []
    strains = {
            "<-3.5%": [],
            "-3.5% to -2.5%": [],
            "-2.5% to -1.5%": [],
            "-1.5% to -0.5%": [],
            "-0.5% to 0.5%": [],
            "0.5% to 1.5%": [],
            "1.5% to 2.5%": [],
            "2.5% to 3.5%": [],
            ">3.5%": [],
            }
    surf_strains = {f"Surface {i}": [] for i in strains.keys()}
    
    ##height percent; measured not in z height, but in
    ##layer the atom belongs to, related to z height, bye the bye
    layers_stepsize = 4 #I dont think we can have >100/4 (i.e. 25) layers in any NP
    strain_layers = {
            f"{i}% to {i+layers_stepsize}%": []
            for i in range(0, 100, layers_stepsize)
            }
    volumetric_strain_layers = {f"Volume {i}": [] for i in strain_layers.keys()}

    df = pd.DataFrame(
            columns=[
                #total Ag count
                "atoms",
                #diameter in nm
                "diameter (nm)",
                #Ag count for the requested system
                "atoms requested",
                #coordination numbers
                "CN1", "CN2", "CN3", "CN4", "CN5", "CN6", "CN7",
                "CN8", "CN9", "CN10", "CN11", "CN12",
                #strains
                "<-3.5%", "-3.5% to -2.5%", "-2.5% to -1.5%",
                "-1.5% to -0.5%", "-0.5% to 0.5%", "0.5% to 1.5%",
                "1.5% to 2.5%", "2.5% to 3.5%", ">3.5%",
                #surface strains
                "Surface <-3.5%", "Surface -3.5% to -2.5%",
                "Surface -2.5% to -1.5%", "Surface -1.5% to -0.5%",
                "Surface -0.5% to 0.5%", "Surface 0.5% to 1.5%",
                "Surface 1.5% to 2.5%", "Surface 2.5% to 3.5%",
                "Surface >3.5%",
                #generalized coordination numbers
                "GCN1", "GCN2", "GCN3", "GCN4", "GCN5", "GCN6",
                "GCN7", "GCN8", "GCN9", "GCN10", "GCN11", "GCN12",
                #surface-to-bulk ratio (in number of Ag)
                "SurfToBulk",
                ], #will include height-varying strains somewhere near the end
            index=range(len(traj))
        )

    ###what kind of system are we analyzing?
    system_type = system_type.lower()
    if system_type not in EXPANDED_TAGS_DICT.keys():
        raise ValueError(
                f"""requested system type ({system_type}) invalid!\n
                Select from:\n
                'all',\n
                'bulk',\n
                'interface',\n
                'perimeter',\n
                'surface',\n
                'surface_perimeter_interface',\n
                OR\n
                'perimeter_interface'
                """
                )
    requested_tag = EXPANDED_TAGS_DICT[system_type]
    if not isinstance(requested_tag, list):
        requested_tag = [requested_tag]
    output_traj: List[Atoms] = []

    ##gather data
    for atoms in tqdm(traj,
            total=len(traj),
            desc="gathering data to record"
            ):
        ##just in case the user submitted substrate-inclusive
        atoms, mgo = divider(atoms)

        ##extract the interested atom indices
        tags = atoms.get_tags()
        wanted_indices = [i in requested_tag for i in tags]
        atoms_filtered = atoms[wanted_indices]
        output_traj.append(atoms_filtered)

        ##run calculations
        CN, CN_matrix, nl = get_CN(
                atoms,
                mult = mult,
                radial_cutoff = radial_cutoff
                )

        GCN = np.round(get_GCN(
                CN = CN,
                CN_matrix = CN_matrix
                )).astype(np.int32)


        strain, surf_strain, normal_strain, layer_percent = compute_strain(
                atoms,
                nl = nl,
                lattice_constant = a,
                system_type = system_type,
                CN = CN,
                )

        ##collect strains
        strains["<-3.5%"].append(
                np.where(strain <= -3.5)[0].shape[0]
                )
        strains[">3.5%"].append(
                np.where(strain > 3.5)[0].shape[0]
                )      
        for key in strains.keys():
            if key not in ["<-3.5%", ">3.5%"]:
                split_key = key.split()
                low = split_key[0]
                low = float(low.split("%")[0])
                high = split_key[-1]
                high = float(high.split("%")[0])
                strains[key].append(np.where(
                    (strain > low) &
                    (strain <= high)
                    )[0].shape[0])


        ##collect surface strains
        surf_strains["Surface <-3.5%"].append(
                np.where(surf_strain <= -3.5)[0].shape[0]
                )
        surf_strains["Surface >3.5%"].append(
                np.where(surf_strain > 3.5)[0].shape[0]
                )
        for key in surf_strains.keys():
            if key not in ["Surface <-3.5%", "Surface >3.5%"]:
                split_key = key.split()
                low = split_key[1]
                low = float(low.split("%")[0])
                high = split_key[-1]
                high = float(high.split("%")[0])
                surf_strains[key].append(np.where(
                    (surf_strain > low) &
                    (surf_strain <= high)
                    )[0].shape[0])


        ##collect layer-varying-averaged normal strains
        for key in strain_layers.keys():
            split_key = key.split()
            low = split_key[0]
            low = float(low.split("%")[0])
            high = split_key[-1]
            high = float(high.split("%")[0])

            if low == 0:
                qualifying_mask = np.where(
                        (layer_percent >= low)
                        &
                        (layer_percent <= high)
                        )[0]
            else:
                qualifying_mask = np.where(
                        (layer_percent > low)
                        &
                        (layer_percent <= high)
                        )[0]

            qualifying_strains = np.array(normal_strain[qualifying_mask])
            strain_layers[key].append(
                    np.nanmean(qualifying_strains) if qualifying_strains.size else np.nan
                    )

        ##collect volumetric strains for each layer
        for key in volumetric_strain_layers.keys():
            split_key = key.split()
            low = split_key[1]
            low = float(low.split("%")[0])
            high = split_key[-1]
            high = float(high.split("%")[0])

            if low == 0:
                qualifying_mask = np.where(
                        (layer_percent >= low)
                        &
                        (layer_percent <= high)
                        )[0]
            else:
                qualifying_mask = np.where(
                        (layer_percent > low)
                        &
                        (layer_percent <= high)
                        )[0]

            qualifying_strains = np.array(strain[qualifying_mask])
            volumetric_strain_layers[key].append(
                    np.nanmean(qualifying_strains) if qualifying_strains.size else np.nan
                    )


        ##collect natoms and surf-to-bulk
        ##!!MIND!! surf here INCLUDES perimeter and interface
        n_atoms.append(len(atoms))
        d_nm = _natoms_to_SCM_footprint_diameter(
                n_atoms = len(atoms),
                element = "Ag",
                a = a,
                theta = theta
                )/10. #nm
        diameter.append(round(d_nm,3))
        r_atoms.append(len(atoms_filtered))
        tags = list(tags)
        n_bulk = tags.count(0)
        n_surf = tags.count(1) + tags.count(2) + tags.count(3)
        try:
            SurfToBulks.append(n_surf / n_bulk)
        except ZeroDivisionError:
            SurfToBulks.append(np.inf)

        ##collect CNs and GCNs
        CN = CN[wanted_indices] #only report CN and GCN for the requested kind of system
        GCN = GCN[wanted_indices]
        for cn in range(1, 13):
            CNs[cn].append(list(CN).count(cn))
            GCNs[cn].append(list(GCN).count(cn))


    #write data
    df.loc[:, "atoms"] = n_atoms #report TOTAL natoms
    df.loc[:, "diameter (nm)"] = diameter #report diameter of TOTAL system (obviously)
    df.loc[:, "atoms requested"] = r_atoms #report natoms of the requested kind of system
    df.loc[:, "SurfToBulk"] = SurfToBulks #same as above

    #strains and surface strains
    for key in strains.keys():
        df.loc[:, key] = strains[key]
    for key in surf_strains.keys():
        df.loc[:, key] = surf_strains[key]

    #height-varying normal strains
    for key in strain_layers.keys():
        df.loc[:, key] = strain_layers[key]
    #height-varying volumetric strains
    for key in volumetric_strain_layers.keys():
        df.loc[:, key] = volumetric_strain_layers[key]


    #CNs and GCNs
    for cn in range(1, 13):
        df.loc[:, f"CN{cn}"] = CNs[cn]
        df.loc[:, f"GCN{cn}"] = GCNs[cn]

    ##manually exclude bulk atoms from CNs if we're not analyzing bulk
    if system_type not in ["bulk", "all"]:
        df.loc[:, "CN12"] = df.loc[:, "GCN12"] = [0] * len(traj)

    output = f"{system_}_{system_type}.csv"
    df.to_csv(output, index=False)
    print(f"{output} CSV written!")

    output_traj_name = f"{system_}_{system_type}.traj"
    write(output_traj_name, output_traj)
    print(f"{output_traj_name} also written!")

    return output_traj


if __name__ == "__main__":
    parser = ArgumentParser(
    description = """For strain and CN analyses\n\n
    Writes the output CSV that will be plotted later\n
    Also writes the traj file with the system_type alone there
    """
    )

    parser.add_argument(
            "--traj", "-t",
            type = str, required = True,
            help = "Traj file on which to run the script. Required"
            )
    parser.add_argument(
            "--system", "-s",
            type = str, required = True,
            help = f"""Name of output file prefix\n
            I would normally expect 'Optimized', 'Wulff', 'Winter', or 'Plato'.\n
            Required"""
            )
    parser.add_argument(
            "--system_type", "-st",
            type = str, required = True,
            choices = list(EXPANDED_TAGS_DICT.keys()),
            help = f"""type of system analyzed, MUST be one of:\n
            'all', 'bulk', 'interface', 'perimeter', 'surface', 'substrate',
            'surface_perimeter_interface',
            'perimeter_interface'\n
            \n
            !!!IF REQUESTING 'surface_perimeter_interface' or 'perimeter_interface',\n
            BE SURE TO WRAP IN QUOTES!!!"""
            )
    parser.add_argument(
            "--mult", "-m",
            type = float, default = MULT,
            help = f"scaling factor for covalent radii. Default = {MULT}"
            )
    parser.add_argument(
            "--lattice", "-l",
            type = float, default = DEFAULT_MLIP_AG_LATTICE,
            help = f"Ag lattice constant. Default = {DEFAULT_MLIP_AG_LATTICE}"
            )
    parser.add_argument(
            "--theta", "-th",
            type = float, default = THETA,
            help = f"Contact angle. Default = {THETA}"
            )

    args = parser.parse_args()
    traj = read(args.traj, ":")
    system_ = args.system
    system_type = args.system_type
    mult = args.mult
    lattice = args.lattice
    theta = args.theta

    _ = write_output(
            traj = traj,
            system_ = system_,
            system_type = system_type,
            mult = mult,
            a = lattice,
            theta = theta,
            )


