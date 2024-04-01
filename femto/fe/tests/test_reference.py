import copy

import mdtraj
import numpy
import openmm.unit
import parmed
import pytest

import femto.fe.reference
from femto.fe.reference import (
    _create_ligand_queries,
    _structure_to_mdtraj,
    queries_to_idxs,
    select_ligand_idxs,
    select_protein_cavity_atoms,
)
from femto.fe.tests.systems import CDK2_SYSTEM
from femto.md.tests.mocking import build_mock_structure


@pytest.fixture
def cdk2_receptor() -> parmed.Structure:
    return parmed.load_file(str(CDK2_SYSTEM.receptor_coords), structure=True)


@pytest.fixture
def cdk2_receptor_traj(cdk2_receptor) -> mdtraj.Trajectory:
    return _structure_to_mdtraj(cdk2_receptor)


@pytest.fixture
def cdk2_ligand_1() -> parmed.amber.AmberParm:
    return parmed.amber.AmberParm(
        str(CDK2_SYSTEM.ligand_1_params), str(CDK2_SYSTEM.ligand_1_coords)
    )


@pytest.fixture
def cdk2_ligand_1_traj(cdk2_ligand_1) -> mdtraj.Trajectory:
    return _structure_to_mdtraj(cdk2_ligand_1)


@pytest.fixture
def cdk2_ligand_1_ref_idxs() -> tuple[int, int, int]:
    # computed using the reference SepTop implementation at commit 7af0b4d
    return 13, 14, 0


@pytest.fixture
def cdk2_ligand_2() -> parmed.amber.AmberParm:
    return parmed.amber.AmberParm(
        str(CDK2_SYSTEM.ligand_2_params), str(CDK2_SYSTEM.ligand_2_coords)
    )


@pytest.fixture
def cdk2_ligand_2_traj(cdk2_ligand_2) -> mdtraj.Trajectory:
    return _structure_to_mdtraj(cdk2_ligand_2)


def test_queries_to_idxs(cdk2_ligand_1):
    actual_idxs = queries_to_idxs(cdk2_ligand_1, ("@11", "@10", "@12"))

    expected_idxs = 10, 9, 11
    assert actual_idxs == expected_idxs


def test_queries_to_idxs_multiple_matched(cdk2_ligand_1):
    with pytest.raises(ValueError, match="atoms while exactly 1 atom was"):
        queries_to_idxs(cdk2_ligand_1, ("@11,12", "@10", "@10"))


def test_create_ligand_queries_chen(cdk2_ligand_1, cdk2_ligand_2):
    ligand_1_ref_masks, ligand_2_ref_masks = _create_ligand_queries(
        cdk2_ligand_1, cdk2_ligand_2, "chen"
    )

    expected_ligand_1_ref_masks = ("@11", "@10", "@12")
    expected_ligand_2_ref_masks = ("@10", "@11", "@25")

    assert ligand_1_ref_masks == expected_ligand_1_ref_masks
    assert ligand_2_ref_masks == expected_ligand_2_ref_masks


def test_create_ligand_queries_collinear(cdk2_ligand_1, mocker):
    mocker.patch("femto.fe.reference._COLLINEAR_THRESHOLD", 1.0 - 1.0e-6)

    subset_1 = copy.deepcopy(cdk2_ligand_1["@/C"][0:4])
    subset_1.coordinates = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.1, 0.0],
    ]

    subset_2 = copy.deepcopy(cdk2_ligand_1["@/C"][0:4])
    subset_2.coordinates = [
        [0.001, 0.0, 0.0],
        [1.002, 0.0, 0.0],
        [2.003, 0.0, 0.0],
        [2.004, 0.1, 0.0],
    ]

    ligand_1_ref_masks, ligand_2_ref_masks = _create_ligand_queries(
        subset_1, subset_2, "chen"
    )

    # @1 @2 @3 would from a straight line so @4 should be chosen over @3
    expected_ligand_1_ref_masks = ("@1", "@2", "@4")
    expected_ligand_2_ref_masks = ("@1", "@2", "@4")

    assert ligand_1_ref_masks == expected_ligand_1_ref_masks
    assert ligand_2_ref_masks == expected_ligand_2_ref_masks


def test_create_ligand_queries_chen_all_co_linear(cdk2_ligand_1):
    subset_1 = copy.deepcopy(cdk2_ligand_1["@/C"][0:3])
    subset_1.coordinates = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]

    subset_2 = copy.deepcopy(cdk2_ligand_1["@/C"][0:3])
    subset_2.coordinates = [[0.001, 0.0, 0.0], [1.002, 0.0, 0.0], [2.003, 0.0, 0.0]]

    with pytest.raises(
        RuntimeError, match="Could not find three non-co-linear reference atoms"
    ):
        _create_ligand_queries(subset_1, subset_2, "chen")


def test_create_ligand_queries_baumann():
    ligand = build_mock_structure(["C1=CC(=CC2=C1CC(=C2)C)C"])

    ref_atoms, _ = _create_ligand_queries(ligand, ligand, "baumann")
    assert ref_atoms == ("@6", "@5", "@9")


def test_create_ligand_queries_required_ligands(cdk2_ligand_1):
    with pytest.raises(ValueError, match="two ligands must be provided"):
        _create_ligand_queries(cdk2_ligand_1, None, "chen")


@pytest.mark.parametrize(
    "ligand_1_queries, expected_ligand_1_idxs",
    [
        (None, (10, 9, 11)),
        (("@1", "@2", "@3"), (0, 1, 2)),
    ],
)
def test_select_ligand_idxs_one_ligand(
    ligand_1_queries, expected_ligand_1_idxs, cdk2_ligand_1, mocker
):
    mock_ligand_1_queries = ("@11", "@10", "@12")

    mocker.patch(
        "femto.fe.reference._create_ligand_queries_baumann",
        return_value=mock_ligand_1_queries,
    )

    expected_method = "baumann"

    ligand_1_idxs, ligand_2_idxs = select_ligand_idxs(
        cdk2_ligand_1, None, expected_method, ligand_1_queries, None
    )

    expected_ligand_1_idxs = expected_ligand_1_idxs
    expected_ligand_2_idxs = None

    assert ligand_1_idxs == expected_ligand_1_idxs
    assert ligand_2_idxs == expected_ligand_2_idxs


@pytest.mark.parametrize(
    "ligand_1_queries, ligand_2_queries, expected_ligand_1_idxs, "
    "expected_ligand_2_idxs",
    [
        (None, None, (10, 9, 11), (9, 10, 24)),
        (None, ("@1", "@2", "@3"), (10, 9, 11), (0, 1, 2)),
        (("@1", "@2", "@3"), None, (0, 1, 2), (9, 10, 24)),
    ],
)
def test_select_ligand_idxs_two_ligands(
    ligand_1_queries,
    ligand_2_queries,
    expected_ligand_1_idxs,
    expected_ligand_2_idxs,
    cdk2_ligand_1,
    cdk2_ligand_2,
    mocker,
):
    mock_ligand_1_queries = ("@11", "@10", "@12")
    mock_ligand_2_queries = ("@10", "@11", "@25")

    mock_create_queries = mocker.patch(
        "femto.fe.reference._create_ligand_queries",
        return_value=(mock_ligand_1_queries, mock_ligand_2_queries),
    )

    expected_method = "chen"

    ligand_1_idxs, ligand_2_idxs = select_ligand_idxs(
        cdk2_ligand_1,
        cdk2_ligand_2,
        expected_method,
        ligand_1_queries,
        ligand_2_queries,
    )
    mock_create_queries.assert_called_once_with(
        cdk2_ligand_1, cdk2_ligand_2, expected_method
    )

    assert ligand_1_idxs == expected_ligand_1_idxs
    assert ligand_2_idxs == expected_ligand_2_idxs


def test_filter_receptor_atoms(
    cdk2_receptor_traj, cdk2_ligand_1_traj, cdk2_ligand_1_ref_idxs
):
    # computed using the reference SepTop implementation at commit 7af0b4d
    # fmt: off
    expected_idxs = [380, 381, 382, 383, 384, 388, 389, 390, 391, 392, 399, 400, 401, 402, 403, 408, 409, 410, 411, 412, 830, 831, 832, 833, 834, 838, 839, 840, 841, 842, 847, 848, 849, 850, 851, 853, 854, 855, 856, 857, 865, 866, 867, 868, 869, 873, 874, 875, 876, 877, 884, 885, 886, 887, 888, 893, 894, 895, 896, 897, 901, 902, 903, 904, 905, 909, 910, 911, 912, 913, 918, 919, 920, 921, 922, 923, 924, 925, 926, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 1494, 1495, 1496, 1497, 1498, 1502, 1503, 1504, 1505, 1506, 1516, 1517, 1518, 1519, 1520, 1522, 1523, 1524, 1525, 1526, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1540, 1541, 1542, 1543, 1544, 1548, 1549, 1550, 1551, 1552, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1691, 1692, 1693, 1694, 1695, 1723, 1734, 2109, 2110, 2111, 2112, 2116, 2117, 2118, 2119, 2120]  # noqa: E201,E221,E241,E501
    # fmt: on

    receptor_idxs = femto.fe.reference._filter_receptor_atoms(
        cdk2_receptor_traj, cdk2_ligand_1_traj, cdk2_ligand_1_ref_idxs[0]
    )
    assert receptor_idxs == expected_idxs


@pytest.mark.parametrize(
    "coords, expected_linear",
    [
        (numpy.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), True),
        (numpy.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.1, 0.0]]), True),
        (numpy.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), True),
        (numpy.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.1, 0.0]]), True),
        (numpy.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), False),
    ],
)
def test_is_angle_linear(coords, expected_linear):
    assert femto.fe.reference._is_angle_linear(coords, (0, 1, 2)) == expected_linear


@pytest.mark.parametrize(
    "coords, expected_trans",
    [
        (
            # \_/
            numpy.array(
                [[-2.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]
            ),
            False,
        ),
        (
            # \-\
            numpy.array(
                [[-2.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, -1.0, 0.0]]
            ),
            True,
        ),
        (
            numpy.array(
                [[-2.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 1.0]]
            ),
            False,
        ),
        (
            numpy.array(
                [[-2.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, -1.0]]
            ),
            False,
        ),
    ],
)
def test_is_dihedral_trans(coords, expected_trans):
    assert femto.fe.reference._is_dihedral_trans(coords, (0, 1, 2, 3)) == expected_trans


@pytest.mark.parametrize(
    "ligand_coords, receptor_coords, "
    "expected_collinear, expected_angle_linear, expected_dihedral_trans, "
    "expected_valid",
    [
        (
            # collinear points
            numpy.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            numpy.array([[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]),
            True,
            None,
            None,
            False,
        ),
        (
            # linear r1,l1,l2 angle but not strictly collinear
            numpy.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]]),
            numpy.array([[-1.0, -0.1, 0.0], [-2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]),
            False,
            True,
            None,
            False,
        ),
        (
            # trans r1,l1,l2,l3 dihedral
            numpy.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 2.0, 0.0]]),
            numpy.array([[-1.0, -1.0, 0.0], [-2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]),
            False,
            False,
            True,
            False,
        ),
        (
            # cis r1,l1,l2,l3 dihedral
            numpy.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 2.0, 0.0]]),
            numpy.array([[-1.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]),
            False,
            False,
            False,
            True,
        ),
    ],
)
def test_is_valid_r1(
    ligand_coords,
    receptor_coords,
    expected_collinear,
    expected_angle_linear,
    expected_dihedral_trans,
    expected_valid,
    mocker,
):
    ligand = build_mock_structure(["O"])
    ligand.coordinates = ligand_coords
    receptor = build_mock_structure(["O"])
    receptor.coordinates = receptor_coords

    mocker.patch("femto.fe.reference._COLLINEAR_THRESHOLD", 1.0 - 1.0e-6)

    spied_collinear = mocker.spy(femto.fe.reference, "_are_collinear")
    spied_linear = mocker.spy(femto.fe.reference, "_is_angle_linear")
    spied_trans = mocker.spy(femto.fe.reference, "_is_dihedral_trans")

    is_valid = femto.fe.reference._is_valid_r1(
        _structure_to_mdtraj(receptor), 0, _structure_to_mdtraj(ligand), (0, 1, 2)
    )
    assert is_valid == expected_valid

    assert spied_collinear.spy_return == expected_collinear
    assert spied_linear.spy_return == expected_angle_linear
    assert spied_trans.spy_return == expected_dihedral_trans


@pytest.mark.parametrize(
    "ligand_coords, receptor_coords, receptor_ref_idx, "
    "expected_collinear, expected_angle_linear, expected_dihedral_trans, "
    "expected_valid",
    [
        (
            # r1 == r2
            numpy.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 0.0, 0.0]]),
            numpy.array([[-1.0, 0.0, 0.0], [-6.0, 10.0, 0.0], [-7.0, 0.0, 0.0]]),
            1,
            None,
            None,
            None,
            False,
        ),
        (
            # r1,r2 too close
            numpy.array([[2.0, 0.0, 0.0], [3.0, 1.0, 0.0], [4.0, 0.0, 0.0]]),
            numpy.array([[1.0, 0.0, 0.0], [-1.0, -0.1, 0.0], [-2.0, 0.0, 0.0]]),
            0,
            None,
            None,
            None,
            False,
        ),
        (
            # collinear points
            numpy.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            numpy.array([[1.0, 0.0, 0.0], [-6.0, 0.0, 0.0], [-7.0, 0.0, 0.0]]),
            0,
            True,
            None,
            None,
            False,
        ),
        (
            # linear r2,r1,l1 angle but not strictly collinear
            numpy.array([[2.0, 0.0, 0.0], [3.0, 1.0, 0.0], [4.0, 0.0, 0.0]]),
            numpy.array([[1.0, 0.0, 0.0], [-6.0, -0.1, 0.0], [-7.0, 0.0, 0.0]]),
            0,
            False,
            True,
            None,
            False,
        ),
        (
            # trans r2,r1,l1,l2 dihedral
            numpy.array([[2.0, 0.0, 0.0], [3.0, 1.0, 0.0], [4.0, 0.0, 0.0]]),
            numpy.array([[1.0, 0.0, 0.0], [-6.0, -10.0, 0.0], [-7.0, 0.0, 0.0]]),
            0,
            False,
            False,
            True,
            False,
        ),
        (
            # cis r2,r1,l1,l2 dihedral
            numpy.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 0.0, 0.0]]),
            numpy.array([[-1.0, 0.0, 0.0], [-6.0, 10.0, 0.0], [-7.0, 0.0, 0.0]]),
            0,
            False,
            False,
            False,
            True,
        ),
    ],
)
def test_is_valid_r2(
    ligand_coords,
    receptor_coords,
    receptor_ref_idx,
    expected_collinear,
    expected_angle_linear,
    expected_dihedral_trans,
    expected_valid,
    mocker,
):
    ligand = build_mock_structure(["O"])
    ligand.coordinates = ligand_coords
    receptor = build_mock_structure(["O"])
    receptor.coordinates = receptor_coords

    mocker.patch("femto.fe.reference._COLLINEAR_THRESHOLD", 1.0 - 1.0e-6)

    spied_collinear = mocker.spy(femto.fe.reference, "_are_collinear")
    spied_linear = mocker.spy(femto.fe.reference, "_is_angle_linear")
    spied_trans = mocker.spy(femto.fe.reference, "_is_dihedral_trans")

    is_valid = femto.fe.reference._is_valid_r2(
        _structure_to_mdtraj(receptor),
        1,
        receptor_ref_idx,
        _structure_to_mdtraj(ligand),
        (0, 1, 2),
    )
    assert is_valid == expected_valid

    assert spied_collinear.spy_return == expected_collinear
    assert spied_linear.spy_return == expected_angle_linear
    assert spied_trans.spy_return == expected_dihedral_trans


@pytest.mark.parametrize(
    "ligand_coords, receptor_coords, receptor_ref_idxs, "
    "expected_collinear, expected_dihedral_trans, expected_valid",
    [
        (
            # r3 == r1
            numpy.array([[2.0, 1.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            numpy.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-6.0, 10.0, 0.0]]),
            (0, 2),
            None,
            None,
            False,
        ),
        (
            # collinear points
            numpy.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            numpy.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            (0, 1),
            True,
            None,
            False,
        ),
        (
            # trans r3,r2,r1,l1 dihedral
            numpy.array([[3.0, 1.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            numpy.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
            (0, 1),
            False,
            True,
            False,
        ),
        (
            # cis r3,r2,r1,l1 dihedral
            numpy.array([[3.0, 1.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            numpy.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            (0, 1),
            False,
            False,
            True,
        ),
    ],
)
def test_is_valid_r3(
    ligand_coords,
    receptor_coords,
    receptor_ref_idxs,
    expected_collinear,
    expected_dihedral_trans,
    expected_valid,
    mocker,
):
    ligand = build_mock_structure(["O"])
    ligand.coordinates = ligand_coords
    receptor = build_mock_structure(["O"])
    receptor.coordinates = receptor_coords

    mocker.patch("femto.fe.reference._COLLINEAR_THRESHOLD", 1.0 - 1.0e-6)

    spied_collinear = mocker.spy(femto.fe.reference, "_are_collinear")
    spied_trans = mocker.spy(femto.fe.reference, "_is_dihedral_trans")

    is_valid = femto.fe.reference._is_valid_r3(
        _structure_to_mdtraj(receptor),
        2,
        receptor_ref_idxs[0],
        receptor_ref_idxs[1],
        _structure_to_mdtraj(ligand),
        (0, 1, 2),
    )
    assert is_valid == expected_valid

    assert spied_collinear.spy_return == expected_collinear
    assert spied_trans.spy_return == expected_dihedral_trans


def test_select_receptor_idxs(cdk2_receptor, cdk2_ligand_1, cdk2_ligand_1_ref_idxs):
    from femto.fe.reference import select_ligand_idxs

    x, _ = select_ligand_idxs(cdk2_ligand_1, None, "baumann")

    # computed using the reference SepTop implementation at commit 7af0b4d
    # note there will be some differences due to the different r3 distance calculation
    # and also the bug with the SepTop implementation of are collinear.
    expected_receptor_idxs = 830, 841, 384

    receptor_idxs = femto.fe.reference.select_receptor_idxs(
        cdk2_receptor, cdk2_ligand_1, cdk2_ligand_1_ref_idxs
    )
    assert receptor_idxs == expected_receptor_idxs


def test_select_protein_cavity_atoms(cdk2_receptor, cdk2_ligand_1, cdk2_ligand_2):
    ligands = [cdk2_ligand_1, cdk2_ligand_2]

    # computed using VMD:
    #
    # (protein + ligands[0] + ligands[1]).save("structure.pdb")
    #
    # set sel [atomselect top "protein and (within 5 of resname MOL) and name CA"]
    # $sel get index
    expected_mask = (
        "@90,98,102,111,145,248,507,640,651,660,671,679,689,698,1059,1068,1163"
    )

    mask = select_protein_cavity_atoms(
        cdk2_receptor, ligands, 5.0 * openmm.unit.angstrom
    )
    assert mask == expected_mask
