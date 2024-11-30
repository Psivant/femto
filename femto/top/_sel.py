"""Select atoms in topologies."""

import abc
import functools
import typing
from urllib.parse import ParseResult

import numpy
import pandas
from pyparsing import (
    Forward,
    Keyword,
    MatchFirst,
    OneOrMore,
    OpAssoc,
    ParseException,
    ParserElement,
    Regex,
    Suppress,
    Word,
    alphanums,
    alphas,
    infix_notation,
    nums,
)

if typing.TYPE_CHECKING:
    import femto.top


ParserElement.enablePackrat()

_IDENTIFIER = Word(alphas, alphanums)

_INTEGER = Word(nums).setParseAction(lambda t: int(t[0]))
_DECIMAL = Regex(r"[+-]?\d+\.\d*").setParseAction(lambda t: float(t[0]))


# fmt: off
_FLAG_OPS = [
    ("all",       ["*"]),
    ("none",      []),
    ("protein",   []),
    ("sidechain", ["sc."]),
    ("backbone",  []),
    ("water",     ["solvent", "sol."])
]
"""Flag selection ops of the form {kw}."""
_ATTR_OPS = [
    ("chain",   ["c."],   _IDENTIFIER),
    ("resn",    ["r."],   _IDENTIFIER),
    ("name",    ["n."],   _IDENTIFIER),
    ("resi",    ["i."],   _INTEGER),
    ("index",   ["idx."], _INTEGER)
]
"""Attribute selection ops of the form {kw} {arg}."""
# fmt: on

_EXPANSION_OPS = [("bychain", ["bc."]), ("byres", ["br."])]

_DIST_X_TO_Y_OPS = [("within", ["w."]), ("near_to", ["nto."]), ("beyond", ["be."])]
"""Distance selection ops of the form S1 {kw} {dist} of S2."""
_DIST_TO_Y_OPS = [("around", ["a."]), ("expand", ["x."])]
"""Distance selection ops of the form S1 {kw} {dist}"""

_UNARY_OPS = [("not", ["!"])]
"""Unary selection ops."""
_BINARY_OPS = [("and", ["&"]), ("or", ["|"])]
"""Binary selection ops."""

_ALL_OPS = (
    _FLAG_OPS
    + _ATTR_OPS
    + _EXPANSION_OPS
    + _DIST_X_TO_Y_OPS
    + _DIST_TO_Y_OPS
    + _UNARY_OPS
    + _BINARY_OPS
)


def _flatten(op: tuple[str, list[str]]) -> tuple[str, ...]:
    """Flatten the keyword and associated aliases of an operation."""
    return op[0], *op[1]


_RESERVED = MatchFirst([Keyword(kw) for op in _ALL_OPS for kw in _flatten(op)])
"""Reserved keywords for the parser."""
_ALIASES = {alias: op[0] for op in _ALL_OPS for alias in _flatten(op)}
"""Aliases for the reserved keywords."""


class BaseOp(abc.ABC):
    @abc.abstractmethod
    def apply(
        self,
        ctx: pandas.DataFrame,
        xyz: numpy.ndarray | None,
        box: numpy.ndarray | None,
    ) -> numpy.ndarray:
        raise NotImplementedError()


class FlagOp(BaseOp):
    """Represents operations that have no arguments, e.g. 'protein', 'backbone'."""

    def __init__(self, tokens):
        kw = _ALIASES[tokens[0]]
        self.kw = kw

    def apply(
        self,
        ctx: pandas.DataFrame,
        xyz: numpy.ndarray | None,
        box: numpy.ndarray | None,
    ) -> numpy.ndarray:
        if self.kw == "all":
            return numpy.ones(len(ctx), dtype=bool)
        elif self.kw == "none":
            return numpy.zeros(len(ctx), dtype=bool)

        return (ctx[self.kw] == True).values.flatten()  # noqa: E712

    def __repr__(self):
        return f"flag(kw={self.kw})"


class AttrOp(BaseOp):
    """Represents operations that filters based on attributes, e.g. 'chain X',
    'atom CA'."""

    def __init__(self, tokens):
        kw, *args = tokens

        self.kw = _ALIASES[kw]
        self.args = {*args}

    def apply(
        self,
        ctx: pandas.DataFrame,
        xyz: numpy.ndarray | None,
        box: numpy.ndarray | None,
    ) -> numpy.ndarray:
        return ctx[self.kw].isin(self.args).values

    def __repr__(self):
        return f"attr(kw={self.kw}, args={self.args})"


class UnaryOp(BaseOp):
    """Represents unary operations, e.g. 'not', 'invert'."""

    def __init__(self, tokens):
        assert len(tokens) == 1, f"expected 1 token, got {len(tokens)}"

        op, self.rhs = tokens[0]
        self.op = _ALIASES[op]

    def apply(
        self,
        ctx: pandas.DataFrame,
        xyz: numpy.ndarray | None,
        box: numpy.ndarray | None,
    ) -> numpy.ndarray:
        if self.op.lower() == "not":
            return ~self.rhs.apply(ctx, xyz, box)

        raise NotImplementedError(f"unsupported unary operation: {self.op}")

    def __repr__(self):
        return f"unary(op='{self.op}', rhs={self.rhs})"


class BinaryOp(BaseOp):
    """Represents binary operations, e.g. 'and', 'or'."""

    def __init__(self, tokens):
        assert len(tokens) == 1, f"expected 1 token, got {len(tokens)}"

        self.lhs, op, self.rhs = tokens[0]
        self.op = _ALIASES[op]

    def apply(
        self,
        ctx: pandas.DataFrame,
        xyz: numpy.ndarray | None,
        box: numpy.ndarray | None,
    ) -> numpy.ndarray:
        lhs = self.lhs.apply(ctx, xyz, box)
        rhs = self.rhs.apply(ctx, xyz, box)

        if self.op.lower() == "and":
            return lhs & rhs
        elif self.op.lower() == "or":
            return lhs | rhs

        raise NotImplementedError(f"unsupported binary operation: {self.op}")

    def __repr__(self):
        return f"compare(op='{self.op}', lhs={self.lhs}, rhs={self.rhs})"


class ExpandOp(BaseOp):
    """Represents expansion operations, e.g. 'byres', 'bychain'."""

    def __init__(self, tokens):
        assert len(tokens) == 1, f"expected 1 token, got {len(tokens)}"

        op, self.rhs = tokens[0]
        self.op = _ALIASES[op]

    def apply(
        self,
        ctx: pandas.DataFrame,
        xyz: numpy.ndarray | None,
        box: numpy.ndarray | None,
    ) -> numpy.ndarray:
        rhs = self.rhs.apply(ctx, xyz, box)

        ctx_rhs = ctx[rhs]

        if self.op.lower() == "byres":
            res_idxs = set(ctx_rhs["_res_idx"].unique())
            return ctx["_res_idx"].isin(res_idxs).values.flatten()
        elif self.op.lower() == "bychain":
            chains = set(ctx_rhs["chain"].unique())
            return ctx["chain"].isin(chains).values.flatten()

        raise NotImplementedError(f"unsupported expansion operation: {self.op}")

    def __repr__(self):
        return f"expand(op='{self.op}', rhs={self.rhs})"


class DistXToYOp(BaseOp):
    """Represents distance operations of the form 'S1 {kw} {dist} of S2'."""

    def __init__(self, tokens):
        assert len(tokens) == 1, f"expected 1 token, got {len(tokens)}"

        self.lhs, op, self.dist, of_kw, self.rhs = tokens[0]
        self.op = _ALIASES[op]

        assert of_kw == "of", f"expected 'of', got {of_kw}"

    def apply(
        self,
        ctx: pandas.DataFrame,
        xyz: numpy.ndarray | None,
        box: numpy.ndarray | None,
    ) -> numpy.ndarray:
        from femto.top._utils import compute_pairwise_distances

        if xyz is None:
            raise ValueError("topology does not have coordinates")

        lhs = self.lhs.apply(ctx, xyz, box)
        rhs = self.rhs.apply(ctx, xyz, box)

        xyz_lhs = xyz[lhs]
        xyz_rhs = xyz[rhs]

        dists = compute_pairwise_distances(xyz_lhs, xyz_rhs, box)

        if self.op.lower() in {"within", "near_to"}:
            mask = (dists < self.dist).any(axis=1)
        else:
            mask = (dists > self.dist).any(axis=1)

        lhs[lhs] &= mask

        if self.op.lower() in {"within", "beyond"}:
            return lhs
        elif self.op.lower() == "near_to":
            lhs[rhs] = False
            return lhs

        raise NotImplementedError(f"unsupported distance operation: {self.op}")

    def __repr__(self):
        return (
            f"dist_x_to_y(select={self.lhs}, op='{self.op}', dist={self.dist}, "
            f"of={self.rhs})"
        )


class DistToYOp(BaseOp):
    """Represents distance operations of the form 'S1 {kw} {dist}'."""

    def __init__(self, tokens):
        assert len(tokens) == 1, f"expected 1 token, got {len(tokens)}"

        self.lhs, op, self.dist = tokens[0]
        self.op = _ALIASES[op]

    def apply(
        self,
        ctx: pandas.DataFrame,
        xyz: numpy.ndarray | None,
        box: numpy.ndarray | None,
    ) -> numpy.ndarray:
        from femto.top._utils import compute_pairwise_distances

        if xyz is None:
            raise ValueError("topology does not have coordinates")

        lhs = self.lhs.apply(ctx, xyz, box)
        xyz_sel = xyz[lhs]

        mask = (compute_pairwise_distances(xyz, xyz_sel, box) < self.dist).any(axis=1)

        if self.op.lower() == "around":
            mask[lhs] = False
            return mask
        elif self.op.lower() == "expand":
            return mask

        raise NotImplementedError(f"unsupported distance operation: {self.op}")

    def __repr__(self):
        return f"dist_to_y(op='{self.op}', of={self.lhs})"


def _create_parser():
    flag_ops = [
        Keyword(kw).setParseAction(FlagOp) for op in _FLAG_OPS for kw in _flatten(op)
    ]
    attr_ops = [
        (Keyword(kw) + OneOrMore(op[2], _RESERVED)).setParseAction(AttrOp)
        for op in _ATTR_OPS
        for kw in _flatten(op)
    ]

    selector_op = MatchFirst(op for op in flag_ops + attr_ops)

    unary_ops = [
        (Keyword(kw), 1, OpAssoc.RIGHT, UnaryOp)
        for op in _UNARY_OPS
        for kw in _flatten(op)
    ]
    binary_ops = [
        (Keyword(kw), 2, OpAssoc.LEFT, BinaryOp)
        for op in _BINARY_OPS
        for kw in _flatten(op)
    ]
    expand_ops = [
        (Keyword(kw), 1, OpAssoc.RIGHT, ExpandOp)
        for op in _EXPANSION_OPS
        for kw in _flatten(op)
    ]

    dist_x_to_y_ops = [
        (Keyword(kw) + _DECIMAL + Keyword("of"), 2, OpAssoc.LEFT, DistXToYOp)
        for op in _DIST_X_TO_Y_OPS
        for kw in _flatten(op)
    ]
    dist_to_y_ops = [
        (Keyword(kw) + _DECIMAL, 1, OpAssoc.LEFT, DistToYOp)
        for op in _DIST_TO_Y_OPS
        for kw in _flatten(op)
    ]

    expr = Forward()
    expr_nested = Suppress("(") + expr + Suppress(")")

    expr <<= infix_notation(
        selector_op | expr_nested,
        unary_ops + binary_ops + expand_ops + dist_to_y_ops + dist_x_to_y_ops,
    )
    return expr


@functools.lru_cache
def _parse_expr(expr: str) -> ParseResult:
    try:
        return _PARSER.parseString(expr, parseAll=True)
    except ParseException as e:
        raise ValueError(f"failed to parse selection: {expr}") from e


_PARSER = _create_parser()


def _is_protein(atom: "femto.top.Atom") -> bool:
    """Check if an atom is part of a protein, based on the residue name."""
    from femto.top._const import AMINO_ACID_CODES

    return atom.residue.name in AMINO_ACID_CODES


def _is_backbone(atom: "femto.top.Atom") -> bool:
    """Check if an atom is part of the protein backbone, based on the atom name."""
    return _is_protein(atom) and (atom.name in {"CA", "C", "O", "N", "OXT", "H"})


def _is_sidechain(atom: "femto.top.Atom") -> bool:
    """Check if an atom is part of the protein sidechain, based on the atom name."""
    return _is_protein(atom) and not _is_backbone(atom)


def _is_water(atom: "femto.top.Atom") -> bool:
    """Check if an atom is part of a water molecule, based on the residue name."""
    from femto.top._const import WATER_RES_NAMES

    return atom.residue.name in WATER_RES_NAMES


def select(topology: "femto.top.Topology", expr: str) -> numpy.ndarray:
    """Select a subset of atoms from a topology based on a Pymol style selection
    expression.

    Args:
        topology: The topology to select atoms from.
        expr: The selection string to parse.

    Returns:
        The indices of the selected atoms.
    """

    parsed = _parse_expr(expr)

    ctx = pandas.DataFrame(
        [
            {
                "protein": _is_protein(atom),
                "sidechain": _is_sidechain(atom),
                "backbone": _is_backbone(atom),
                "water": _is_water(atom),
                "chain": atom.chain.id,
                "resn": atom.residue.name,
                "name": atom.name,
                "resi": atom.residue.seq_num,
                "index": atom.index + 1,
                "_res_idx": atom.residue.index,
            }
            for atom in topology.atoms
        ]
    )

    selected: numpy.ndarray = parsed[0].apply(ctx, topology.xyz, topology.box)

    idxs = numpy.where(selected)[0]
    return idxs
