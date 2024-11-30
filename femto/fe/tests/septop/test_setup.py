import copy

import numpy
import openmm.app
import openmm.unit
import pytest

import femto.fe.config
import femto.fe.septop
import femto.md.config
import femto.md.prepare
import femto.md.utils.openmm
import femto.top
from femto.fe.tests.systems import CDK2_SYSTEM
from femto.md.tests.mocking import build_mock_structure


@pytest.fixture
def mock_setup_config() -> femto.fe.septop.SepTopSetupStage:
    return femto.fe.septop.SepTopSetupStage(
        restraints=femto.fe.septop.DEFAULT_SOLUTION_RESTRAINTS
    )


@pytest.fixture
def cdk2_ligand_1() -> femto.top.Topology:
    return femto.md.prepare.load_ligand(CDK2_SYSTEM.ligand_1_coords)


@pytest.fixture
def cdk2_ligand_2() -> femto.top.Topology:
    return femto.md.prepare.load_ligand(CDK2_SYSTEM.ligand_2_coords)


@pytest.fixture
def cdk2_receptor() -> femto.top.Topology:
    return femto.md.prepare.load_receptor(CDK2_SYSTEM.receptor_coords)


def test_compute_ligand_offset():
    ligand_1 = build_mock_structure(["[H]Cl"])
    ligand_1.xyz = (
        numpy.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) * openmm.unit.angstrom
    )
    ligand_2 = build_mock_structure(["[H]Cl"])
    ligand_2.xyz = (
        numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]) * openmm.unit.angstrom
    )

    expected_distance = (2.0 + 1.0) * 1.5
    expected_offset = numpy.array([expected_distance, 0.0, -1.0]) * openmm.unit.angstrom

    offset = femto.fe.septop._setup._compute_ligand_offset(ligand_1, ligand_2)

    assert femto.md.utils.openmm.all_close(offset, expected_offset)


def test_apply_complex_restraints(mocker):
    expected_distance = 10.0
    expected_scale = 4.0  # scale should be (10.0 / 5.0)^2

    receptor = build_mock_structure(["[Ar]"])

    ligand_1 = build_mock_structure(["[Ar]"])
    ligand_1.xyz = numpy.array([[expected_distance, 0.0, 0.0]]) * openmm.unit.angstrom

    expected_name = "test restraint"

    mock_force = openmm.CustomCompoundBondForce(0, "")
    mock_force.setName(expected_name)

    mock_create_restraint = mocker.patch(
        "femto.md.restraints.create_boresch_restraint",
        autospec=True,
        return_value=mock_force,
    )

    system = openmm.System()

    config = copy.deepcopy(femto.fe.septop.DEFAULT_COMPLEX_RESTRAINTS)
    config.k_angle_a = 1.0 * openmm.unit.kilocalorie_per_mole / openmm.unit.radians**2

    femto.fe.septop._setup._apply_complex_restraints(
        receptor + ligand_1,
        (0, 0, 0),
        (1, 1, 1),
        config,
        system,
        femto.fe.septop._setup.LAMBDA_BORESCH_LIGAND_1,
    )

    assert system.getNumForces() == 1

    force = system.getForce(0)
    assert force.getName() == expected_name

    mock_create_restraint.assert_called_once_with(
        mocker.ANY, (0, 0, 0), (1, 1, 1), mocker.ANY, "lambda_boresch_lig_1"
    )

    scaled_config = mock_create_restraint.call_args.args[0]
    assert scaled_config.k_angle_a.value_in_unit(
        openmm.unit.kilocalorie_per_mole / openmm.unit.radian**2
    ) == pytest.approx(expected_scale)


def test_apply_solution_restraints():
    expected_distance = 10.0

    ligand_1 = build_mock_structure(["[Ar]"])
    ligand_2 = build_mock_structure(["[Ar]"])
    ligand_2.xyz = numpy.array([[expected_distance, 0.0, 0.0]]) * openmm.unit.angstrom

    system = openmm.System()

    config = copy.deepcopy(femto.fe.septop.DEFAULT_SOLUTION_RESTRAINTS)

    femto.fe.septop._setup._apply_solution_restraints(
        ligand_1 + ligand_2, 0, 1, config, system
    )

    assert system.getNumForces() == 1

    force = system.getForce(0)
    assert isinstance(force, openmm.HarmonicBondForce)
    assert force.getNumBonds() == 1

    idx_1, idx_2, length, k = force.getBondParameters(0)

    assert idx_1 == 0
    assert idx_2 == 1

    assert femto.md.utils.openmm.is_close(
        length, expected_distance * openmm.unit.angstrom
    )
    assert femto.md.utils.openmm.is_close(k, config.k_distance)


def test_setup_system_abfe(cdk2_ligand_1, cdk2_receptor, mock_setup_config, mocker):
    n_ligand_atoms = len(cdk2_ligand_1.atoms)

    def mock_prepare_system(receptor, lig_1, lig_2, *_, **__):
        assert lig_2 is None

        lig_1.residues[0].name = "L1"

        bound = lig_1 + receptor
        bound.box = numpy.eye(3) * 100.0 * openmm.unit.angstrom

        mock_system = openmm.System()

        for _ in range(bound.n_atoms):
            mock_system.addParticle(1.0 * openmm.unit.amu)

        return bound, mock_system

    mock_prepare = mocker.patch(
        "femto.md.prepare.prepare_system",
        autospec=True,
        side_effect=mock_prepare_system,
    )

    mock_apply_hmr = mocker.patch("femto.md.prepare.apply_hmr", autospec=True)
    mock_apply_rest = mocker.patch("femto.md.rest.apply_rest", autospec=True)
    mock_apply_fep = mocker.patch("femto.fe.fep.apply_fep", autospec=True)

    mock_setup_config.apply_rest = True

    expected_h_mass = 2.0 * openmm.unit.amu
    mock_setup_config.hydrogen_mass = expected_h_mass

    (
        system,
        topology,
        ligand_1_ref_idxs,
        ligand_2_ref_idxs,
    ) = femto.fe.septop._setup._setup_system(
        mock_setup_config,
        cdk2_ligand_1,
        None,
        cdk2_receptor,
        [],
        ligand_1_ref_query=None,
        ligand_2_ref_query=None,
        ligand_2_offset=None,
        extra_params=None,
    )

    assert isinstance(topology, femto.top.Topology)
    assert isinstance(system, openmm.System)

    expected_ligand_idxs = set(range(n_ligand_atoms))

    mock_prepare.assert_called_once()
    mock_apply_hmr.assert_called_once_with(mocker.ANY, mocker.ANY, expected_h_mass)

    mock_apply_rest.assert_called_once_with(
        mocker.ANY, expected_ligand_idxs, mock_setup_config.rest_config
    )
    mock_apply_fep.assert_called_once_with(
        mocker.ANY, expected_ligand_idxs, set(), mock_setup_config.fep_config
    )

    assert len(topology.atoms) == system.getNumParticles()


def test_setup_system_rbfe(
    cdk2_ligand_1, cdk2_ligand_2, cdk2_receptor, mock_setup_config, mocker
):
    n_ligand_1_atoms = len(cdk2_ligand_1.atoms)
    n_ligand_2_atoms = len(cdk2_ligand_2.atoms)

    def mock_prepare_system(receptor, lig_1, lig_2, *_, **__):
        lig_1.residues[0].name = "L1"
        lig_2.residues[0].name = "R1"

        bound = lig_1 + lig_2 + receptor
        bound.box = numpy.eye(3) * 100.0 * openmm.unit.angstrom

        mock_system = openmm.System()

        for _ in range(bound.n_atoms):
            mock_system.addParticle(1.0 * openmm.unit.amu)

        return bound, mock_system

    mock_prepare = mocker.patch(
        "femto.md.prepare.prepare_system",
        autospec=True,
        side_effect=mock_prepare_system,
    )

    mocker.patch("femto.md.prepare.apply_hmr", autospec=True)
    mock_apply_rest = mocker.patch("femto.md.rest.apply_rest", autospec=True)
    mock_apply_fep = mocker.patch("femto.fe.fep.apply_fep", autospec=True)

    expected_ligand_1_ref_idxs = 3, 4, 5
    expected_ligand_2_ref_idxs = 8, 9, 10

    mocker.patch(
        "femto.fe.reference.select_ligand_idxs",
        autospec=True,
        return_value=(expected_ligand_1_ref_idxs, expected_ligand_2_ref_idxs),
    )

    mock_setup_config.apply_rest = True

    (
        system,
        topology,
        ligand_1_ref_idxs,
        ligand_2_ref_idxs,
    ) = femto.fe.septop._setup._setup_system(
        mock_setup_config,
        cdk2_ligand_1,
        cdk2_ligand_2,
        cdk2_receptor,
        [],
        ligand_1_ref_query=None,
        ligand_2_ref_query=None,
        ligand_2_offset=None,
        extra_params=None,
    )

    assert isinstance(topology, femto.top.Topology)
    assert isinstance(system, openmm.System)

    expected_ligand_1_idxs = set(range(n_ligand_1_atoms))
    expected_ligand_2_idxs = {i + n_ligand_1_atoms for i in range(n_ligand_2_atoms)}

    mock_prepare.assert_called_once()

    mock_apply_rest.assert_called_once_with(
        mocker.ANY,
        {*expected_ligand_1_idxs, *expected_ligand_2_idxs},
        mock_setup_config.rest_config,
    )
    mock_apply_fep.assert_called_once_with(
        mocker.ANY,
        expected_ligand_1_idxs,
        expected_ligand_2_idxs,
        mock_setup_config.fep_config,
    )

    assert ligand_1_ref_idxs == expected_ligand_1_ref_idxs
    assert ligand_2_ref_idxs == (
        expected_ligand_2_ref_idxs[0] + n_ligand_1_atoms,
        expected_ligand_2_ref_idxs[1] + n_ligand_1_atoms,
        expected_ligand_2_ref_idxs[2] + n_ligand_1_atoms,
    )

    assert len(topology.atoms) == system.getNumParticles()


def test_setup_complex(cdk2_ligand_1, cdk2_ligand_2, cdk2_receptor, mocker):
    n_ligand_atoms = len(cdk2_ligand_1.atoms) + len(cdk2_ligand_2.atoms)

    cdk2_ligand_1.residues[0].name = "L1"
    cdk2_ligand_2.residues[0].name = "R1"

    topology = cdk2_ligand_1 + cdk2_ligand_2 + cdk2_receptor

    mocker.patch(
        "femto.fe.reference.queries_to_idxs",
        autospec=True,
        return_value=(0, 1, 2),
    )

    mock_setup_system = mocker.patch(
        "femto.fe.septop._setup._setup_system",
        autospec=True,
        return_value=(openmm.System(), topology, (0, 1, 2), (3, 4, 5)),
    )
    mock_apply_restraints = mocker.patch(
        "femto.fe.septop._setup._apply_complex_restraints", autospec=True
    )

    mock_config = femto.fe.septop.SepTopSetupStage(
        restraints=femto.fe.septop.DEFAULT_COMPLEX_RESTRAINTS
    )
    femto.fe.septop._setup.setup_complex(
        mock_config,
        cdk2_receptor,
        cdk2_ligand_1,
        cdk2_ligand_2,
        [],
        ("@1", "@2", "@3"),
        ligand_1_ref_query=None,
        ligand_2_ref_query=None,
    )

    mock_setup_system.assert_called_once_with(
        mock_config,
        cdk2_ligand_1,
        cdk2_ligand_2,
        cdk2_receptor,
        [],
        None,
        None,
        None,
        None,
    )
    mock_apply_restraints.assert_has_calls(
        [
            mocker.call(
                mocker.ANY,
                (n_ligand_atoms, n_ligand_atoms + 1, n_ligand_atoms + 2),
                (0, 1, 2),
                mock_config.restraints,
                mocker.ANY,
                femto.fe.septop._setup.LAMBDA_BORESCH_LIGAND_1,
            ),
            mocker.call(
                mocker.ANY,
                (n_ligand_atoms, n_ligand_atoms + 1, n_ligand_atoms + 2),
                (3, 4, 5),
                mock_config.restraints,
                mocker.ANY,
                femto.fe.septop._setup.LAMBDA_BORESCH_LIGAND_2,
            ),
        ]
    )


def test_setup_solution(cdk2_ligand_1, cdk2_ligand_2, mock_setup_config, mocker):
    expected_offset = numpy.array([1.0, 0.0, 0.0]) * openmm.unit.angstrom

    mocker.patch(
        "femto.fe.septop._setup._compute_ligand_offset",
        autospec=True,
        return_value=expected_offset,
    )

    mock_setup_system = mocker.patch(
        "femto.fe.septop._setup._setup_system",
        autospec=True,
        return_value=(openmm.System(), mocker.MagicMock(), (0, 1, 2), (3, 4, 5)),
    )
    mock_apply_restraints = mocker.patch(
        "femto.fe.septop._setup._apply_solution_restraints", autospec=True
    )

    femto.fe.septop._setup.setup_solution(
        mock_setup_config,
        cdk2_ligand_1,
        cdk2_ligand_2,
        ligand_1_ref_query=None,
        ligand_2_ref_query=None,
    )

    mock_setup_system.assert_called_once_with(
        mock_setup_config,
        cdk2_ligand_1,
        cdk2_ligand_2,
        None,
        None,
        None,
        None,
        pytest.approx(-expected_offset),
        None,
    )
    mock_apply_restraints.assert_called_once_with(
        mocker.ANY, 1, 4, mock_setup_config.restraints, mocker.ANY
    )
