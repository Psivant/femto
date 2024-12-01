import numpy
import openmm.unit
import pytest
from openmm.app import PDBFile

from femto.top import Topology
from femto.top._sel import (
    AttrOp,
    BinaryOp,
    DistToYOp,
    DistXToYOp,
    ExpandOp,
    FlagOp,
    UnaryOp,
    select,
)

MOCK_PDB = """CRYST1   73.990  134.730  148.120  90.00  90.00  90.00 P 1
HETATM    2  CH3 ACE A   1      16.168  62.929  36.845  0.00  0.00           C
HETATM    5  C   ACE A   1      16.053  62.130  38.139  0.00  0.00           C
HETATM    6  O   ACE A   1      16.820  61.196  38.364  0.00  0.00           O
ATOM      7  N   SER A   2      15.119  62.551  39.005  0.00  0.00           N
ATOM      9  CA  SER A   2      15.046  62.141  40.409  0.00  0.00           C
ATOM     11  CB  SER A   2      14.249  63.205  41.186  0.00  0.00           C
ATOM     14  OG  SER A   2      14.314  62.985  42.579  0.00  0.00           O
ATOM     16  C   SER A   2      14.522  60.706  40.612  0.00  0.00           C
ATOM     17  O   SER A   2      15.265  59.866  41.118  0.00  0.00           O
ATOM     18  N   MET B   3      13.274  60.444  40.184  0.00  0.00           N
ATOM     20  CA  MET B   3      12.691  59.097  40.113  0.00  0.00           C
ATOM     22  CB  MET B   3      11.246  59.107  40.663  0.00  0.00           C
ATOM     25  CG  MET B   3      11.134  59.461  42.156  0.00  0.00           C
ATOM     28  SD  MET B   3      12.037  58.387  43.308  0.00  0.00           S
ATOM     29  CE  MET B   3      11.191  56.805  43.078  0.00  0.00           C
ATOM     33  C   MET B   3      12.759  58.523  38.684  0.00  0.00           C
ATOM     34  O   MET B   3      12.101  57.516  38.421  0.00  0.00           O
HETATM   35  O   HOH C   1      16.168  62.929  36.845  0.00  0.00           O
HETATM   36  NA  NA  D   1      16.168  62.929  36.845  0.00  0.00          Na+
"""


@pytest.fixture
def mock_top(tmp_path):
    (tmp_path / "mock.pdb").write_text(MOCK_PDB)
    pdb = PDBFile(str(tmp_path / "mock.pdb"))

    top = Topology.from_openmm(pdb.topology)

    xyz = pdb.positions.value_in_unit(openmm.unit.angstrom)
    top.xyz = numpy.array(xyz) * openmm.unit.angstrom

    return top


def test_parse_pad_brackets(mock_top):
    with pytest.raises(ValueError, match="failed to parse selection"):
        select(mock_top, "(name O) and name C)")


@pytest.mark.parametrize(
    "expr, expected_sel",
    [
        ("all", numpy.arange(19)),
        ("none", numpy.array([])),
        ("protein", numpy.arange(17)),
        ("sidechain", numpy.array([0, 5, 6, 11, 12, 13, 14])),
        ("sc.", numpy.array([0, 5, 6, 11, 12, 13, 14])),
        ("backbone", numpy.array([1, 2, 3, 4, 7, 8, 9, 10, 15, 16])),
        ("water", numpy.array([17])),
        ("ion", numpy.array([18])),
        ("chain B", numpy.array([9, 10, 11, 12, 13, 14, 15, 16])),
        ("c. B", numpy.array([9, 10, 11, 12, 13, 14, 15, 16])),
        ("resn MET", numpy.array([9, 10, 11, 12, 13, 14, 15, 16])),
        ("r. MET", numpy.array([9, 10, 11, 12, 13, 14, 15, 16])),
        ("name CA", numpy.array([4, 10])),
        ("n. CA", numpy.array([4, 10])),
        ("elem C+Na", numpy.array([0, 1, 4, 5, 7, 10, 11, 12, 14, 15, 18])),
        ("e. C+Na", numpy.array([0, 1, 4, 5, 7, 10, 11, 12, 14, 15, 18])),
        ("resi 2", numpy.array([3, 4, 5, 6, 7, 8])),
        ("i. 2", numpy.array([3, 4, 5, 6, 7, 8])),
        ("index 2", numpy.array([1])),
        ("idx. 2", numpy.array([1])),
        ("byres idx. 1", numpy.array([0, 1, 2])),
        ("br. idx. 1", numpy.array([0, 1, 2])),
        ("br. name CA", numpy.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])),
        ("bychain idx. 1", numpy.arange(9)),
        ("bc. idx. 1", numpy.arange(9)),
        ("bc. name CA", numpy.arange(17)),
        ("e. Na or r. ACE or r. HOH", numpy.array([0, 1, 2, 17, 18])),
        ("r. ACE | r. HOH", numpy.array([0, 1, 2, 17])),
        ("r. ACE and r. HOH", numpy.array([])),
        ("r. ACE and not r. HOH", numpy.array([0, 1, 2])),
        ("r. ACE &   not r. HOH", numpy.array([0, 1, 2])),
        ("r. ACE and ! r. HOH", numpy.array([0, 1, 2])),
        ("r. ACE and (not r. HOH)", numpy.array([0, 1, 2])),
        ("idx. 1+2+3-5+7-9", numpy.array([0, 1, 2, 3, 4, 6, 7, 8])),
        ("n. CA+CB", numpy.array([4, 5, 10, 11])),
    ],
)
def test_select(mock_top, expr, expected_sel):
    sel = select(mock_top, expr)
    assert numpy.allclose(sel, expected_sel)


@pytest.mark.parametrize(
    "expr, expected_sel",
    [
        ("name O around 0.75", numpy.array([2])),
        ("name O a.     0.75", numpy.array([2])),
        ("name O expand 0.75", numpy.array([1, 2])),
        ("name O x.     0.75", numpy.array([1, 2])),
    ],
)
def test_dist_to_y(expr: str, expected_sel: numpy.ndarray):
    top = Topology()

    chain = top.add_chain("A")
    res = top.add_residue("UNK", 1, chain)

    top.add_atom("C", 0, 0, 1, res)
    top.add_atom("O", 0, 0, 1, res)
    top.add_atom("N", 0, 0, 1, res)

    top.xyz = numpy.array([[0, 0, 0], [0, 0, 1], [0, 0, 1.5]])
    top.box = numpy.eye(3) * 10.0

    sel = select(top, expr)
    assert numpy.allclose(sel, expected_sel)


@pytest.mark.parametrize(
    "expr, expected_sel",
    [
        ("name C or name O within  0.75 of name O", numpy.array([1, 3])),
        ("name C or name O w.      0.75 of name O", numpy.array([1, 3])),
        ("name C or name O near_to 0.75 of name O", numpy.array([3])),
        ("name C or name O nto.    0.75 of name O", numpy.array([3])),
        ("* beyond 0.55 of name O", numpy.array([0, 3])),
        ("* be.    0.55 of name O", numpy.array([0, 3])),
    ],
)
def test_x_dist_to_y(expr: str, expected_sel: numpy.ndarray):
    top = Topology()

    chain = top.add_chain("A")
    res = top.add_residue("UNK", 1, chain)

    top.add_atom("C", 0, 0, 1, res)
    top.add_atom("O", 0, 0, 1, res)
    top.add_atom("N", 0, 0, 1, res)
    top.add_atom("C", 0, 0, 1, res)

    top.xyz = numpy.array([[0, 0, 0], [0, 0, 1], [0, 0, 1.5], [0, 0, 1.6]])
    top.box = numpy.eye(3) * 10.0

    sel = select(top, expr)
    assert numpy.allclose(sel, expected_sel)


@pytest.mark.parametrize(
    "val, expected",
    [
        (FlagOp(["all"]), "flag(kw=all)"),
        (AttrOp(["name", "CA"]), "attr(kw=name, args={'CA'})"),
        (UnaryOp([["not", "rhs_val"]]), "unary(op='not', rhs=rhs_val)"),
        (
            BinaryOp([["lhs_val", "and", "rhs_val"]]),
            "compare(op='and', matchers=['lhs_val', 'rhs_val'])",
        ),
        (ExpandOp([["byres", "rhs_val"]]), "expand(op='byres', rhs=rhs_val)"),
        (
            DistXToYOp([["lhs_val", "within", 1.23, "of", "rhs_val"]]),
            "dist_x_to_y(select=lhs_val, op='within', dist=1.23, of=rhs_val)",
        ),
        (
            DistToYOp([["lhs_val", "around", 1.23]]),
            "dist_to_y(op='around', of=lhs_val)",
        ),
    ],
)
def test_repr(val, expected):
    assert repr(val) == expected
