import openmm
import openmm.unit
import parmed

from femto.fe.tests.systems import CDK2_SYSTEM, TEMOA_SYSTEM
from femto.md.constants import LIGAND_1_RESIDUE_NAME, LIGAND_2_RESIDUE_NAME
from femto.md.system import apply_hmr, load_ligand, load_ligands, load_receptor
from femto.md.tests.mocking import build_mock_structure
from femto.md.utils.openmm import is_close


def test_hmr():
    topology = build_mock_structure(["CC"])

    system = openmm.System()
    system.addParticle(12.0 * openmm.unit.amu)
    system.addParticle(12.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)

    original_mass = sum(
        [system.getParticleMass(i) for i in range(system.getNumParticles())],
        0.0 * openmm.unit.amu,
    )

    expected_h_mass = 1.5 * openmm.unit.amu
    apply_hmr(system, topology, hydrogen_mass=expected_h_mass)

    new_masses = [system.getParticleMass(i) for i in range(system.getNumParticles())]
    new_mass = sum(new_masses, 0.0 * openmm.unit.amu)

    assert is_close(new_mass, original_mass)

    expected_masses = [
        (12.0 - 0.5 * 3.0) * openmm.unit.amu,
        (12.0 - 0.5 * 3.0) * openmm.unit.amu,
    ] + ([expected_h_mass] * 6)

    assert all(
        is_close(new_mass, expected_mass)
        for new_mass, expected_mass in zip(new_masses, expected_masses, strict=True)
    )


def test_hmr_water():
    """HMR should not modify water molecules."""
    topology = build_mock_structure(["O"])

    expected_masses = [16.0, 1.0, 1.0] * openmm.unit.amu

    system = openmm.System()

    for mass in expected_masses:
        system.addParticle(mass)

    apply_hmr(system, topology)

    new_masses = [system.getParticleMass(i) for i in range(system.getNumParticles())]

    assert all(
        is_close(new_mass, expected_mass)
        for new_mass, expected_mass in zip(new_masses, expected_masses, strict=True)
    )


def test_load_ligand():
    reside_name = "ABC"

    ligand = load_ligand(
        CDK2_SYSTEM.ligand_1_coords, CDK2_SYSTEM.ligand_1_params, reside_name
    )

    assert len(ligand[":ABC"].atoms) == len(ligand.atoms)


def test_load_ligands():
    coord_path = CDK2_SYSTEM.ligand_1_coords
    param_path = CDK2_SYSTEM.ligand_1_params

    ligand_1, ligand_2 = load_ligands(coord_path, param_path, None, None)
    assert ligand_2 is None

    assert ligand_1.residues[0].name == LIGAND_1_RESIDUE_NAME

    ligand_1, ligand_2 = load_ligands(coord_path, param_path, coord_path, param_path)
    assert isinstance(ligand_2, parmed.Structure)

    assert ligand_1.residues[0].name == LIGAND_1_RESIDUE_NAME
    assert ligand_2.residues[0].name == LIGAND_2_RESIDUE_NAME


def test_load_receptor_with_params(mocker):
    mock_parameterize = mocker.patch(
        "femto.md.utils.amber.parameterize_structure", autospec=True
    )

    receptor = load_receptor(TEMOA_SYSTEM.receptor_coords, TEMOA_SYSTEM.receptor_params)
    assert isinstance(receptor, parmed.Structure)
    assert receptor.residues[0].name == "<0>"

    assert mock_parameterize.call_count == 0


def test_load_receptor_without_params(mocker):
    mock_parameterize = mocker.patch(
        "femto.md.utils.amber.parameterize_structure", autospec=True
    )
    mock_sources = ["source1"]

    load_receptor(CDK2_SYSTEM.receptor_coords, None, mock_sources)
    mock_parameterize.assert_called_once_with(mocker.ANY, mock_sources)
