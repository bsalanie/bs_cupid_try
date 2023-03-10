"""This module contains some utility programs used by the package."""

import sys
from math import exp, log, sqrt
from traceback import extract_stack
from typing import (
    Any,
    Callable,
    Final,
    Literal,
    NamedTuple,
    Optional,
    TypeAlias,
    overload,
)

import numpy as np
import scipy.stats as sts

TwoArrays: TypeAlias = tuple[np.ndarray, np.ndarray]
ThreeArrays: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]
FourArrays: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

Nest: TypeAlias = list[int]
NestsList: TypeAlias = list[Nest]

ScalarFunctionAndGradient: TypeAlias = Callable[
    [np.ndarray, list, Optional[bool]], float | tuple[float, np.ndarray]
]

"""Type of f(v, args, gr) that returns a scalar value
and also a gradient if gr is True"""

# for numerical derivatives
_EPS: Final = 1e-4
_TWO_EPS: Final = 2.0 * _EPS


def bs_name_func(back: int = 2) -> str:
    """Get the name of the current function, or further back in the stack

    Args:
        back: 2 for the current function, 3 for the function that called it, etc

    Returns:
        the name of the function requested
    """
    stack = extract_stack()
    func_name: str = stack[-back][2]
    return func_name


def print_stars(title: Optional[str] = None, n: int = 70) -> None:
    """Prints a starred line, or two around the title

    Args:
        title:  an optional title
        n: the number of stars on the line

    Returns:
        nothing
    """
    line_stars = "*" * n
    print()
    print(line_stars)
    if title:
        print(title.center(n))
        print(line_stars)
    print()


def bs_error_abort(msg: str = "error, aborting") -> None:
    """Report error and exits with code 1

    Args:
        msg: specifies the error message

    Returns:
        nothing
    """
    print_stars(f"{bs_name_func(3)}: {msg}")
    sys.exit(1)


def test_vector(x: Any, fun_name: Optional[str] = None) -> int:
    """Tests that `x` is a vector; aborts otherwise

    Args:
        x: a potential vector
        fun_name: the name of the calling function

    Returns:
        the size of `x` if it is a vector
    """
    fun_str = ["" if fun_name is None else fun_name + ":"]
    if not isinstance(x, np.ndarray):
        bs_error_abort(f"{fun_str} x should be a Numpy array")
    ndims_x = x.ndim
    if ndims_x != 1:
        bs_error_abort(
            f"{fun_str} x should have one dimension, not {ndims_x}"
        )
    sx: int = x.size
    return sx


def test_matrix(x: Any, fun_name: Optional[str] = None) -> tuple[int, int]:
    """Tests that `x` is a matrix; aborts otherwise

    Args:
        x: a potential matrix
        fun_name: the name of the calling function

    Returns:
        the shape of `x` if it is a matrix
    """
    fun_str: str = "" if fun_name is None else fun_name + ":"
    if not isinstance(x, np.ndarray):
        bs_error_abort(f"x in {fun_str} should be a Numpy array")
    ndims_x = x.ndim
    if ndims_x != 2:
        bs_error_abort(
            f"x in {fun_str} should have two dimensions, not {ndims_x}"
        )
    shx: tuple[int, int] = x.shape
    return shx


def describe_array(v: np.ndarray, name: str = "The array") -> NamedTuple:
    """Descriptive statistics on an array interpreted as a vector

    Args:
        v: the array
        name: its name

    Returns:
        a `DescribeResult` namedtuple
    """
    print_stars(f"{name} has:")
    d = sts.describe(v, None)
    print(f"Number of elements: {d.nobs}")
    print(f"Minimum: {d.minmax[0]}")
    print(f"Maximum: {d.minmax[1]}")
    print(f"Mean: {d.mean}")
    print(f"Stderr: {sqrt(d.variance)}")
    return d


def nprepeat_col(v: np.ndarray, n: int) -> np.ndarray:
    """Creates a matrix with `n` columns, all equal to `v`

    Args:
        v: a vector of size `m`
        n: the number of columns requested

    :return: a matrix of shape `(m, n)`
    """
    _ = test_vector(v, "nprepeat_col")
    w: np.ndarray = v[:, np.newaxis]
    return np.repeat(w, n, axis=1)


def nprepeat_row(v: np.ndarray, m: int) -> np.ndarray:
    """
    Creates a matrix with `m` rows, all equal to `v`

    Args:
        v: a vector of size `n`
        m: the number of rows requested

    Returns:
        a matrix of shape `(m, n)`
    """
    _ = test_vector(v, "nprepeat_row")
    return np.repeat(v[np.newaxis, :], m, axis=0)


def npmaxabs(a: np.ndarray) -> float:
    """The maximum absolute value in an array

    Args:
        a: the array

    Returns:
        $\max{\vert a \vert}$
    """
    maxi: float = np.max(np.abs(a))
    return maxi


@overload
def nplog(a: np.ndarray) -> np.ndarray:
    ...


@overload
def nplog(
    a: np.ndarray,
    deriv: Literal[False],
    eps: float,
    verbose: bool,
) -> np.ndarray:
    ...


@overload
def nplog(
    a: np.ndarray,
    deriv: Literal[True],
    eps: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    ...


def nplog(
    a: np.ndarray,
    deriv: bool = False,
    eps: float = 1e-30,
    verbose: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """$C^2$ extension of  $\ln(a)$ below `eps`

    Args:
        a: a Numpy array
        deriv: if `True`, the first derivative is also returned
        eps: a lower bound
        verbose: whether diagnoses are printed

    Returns:
        $\ln(a)$ $C^2$-extended below `eps`,
        with its derivative if `deriv` is `True`
    """
    if np.min(a) > eps:
        loga = np.log(a)
        return (loga, 1.0 / a) if deriv else loga
    else:
        logarreps = np.log(np.maximum(a, eps))
        logarr_smaller = log(eps) - (eps - a) * (3.0 * eps - a) / (
            2.0 * eps * eps
        )
        if verbose:
            n_small_args = np.sum(a < eps)
            if n_small_args > 0:
                finals = "s" if n_small_args > 1 else ""
                print(
                    f"nplog: {n_small_args} argument{finals} smaller than {eps}: mini = {np.min(a)}"
                )
        loga = np.where(a > eps, logarreps, logarr_smaller)
        if deriv:
            der_logarreps = 1.0 / np.maximum(a, eps)
            der_logarr_smaller = (2.0 * eps - a) / (eps * eps)
            der_loga = np.where(a > eps, der_logarreps, der_logarr_smaller)
            return loga, der_loga
        else:
            return loga


@overload
def npexp(
    a: np.ndarray,
    deriv: Literal[False],
) -> np.ndarray:
    ...


@overload
def npexp(
    a: np.ndarray,
    deriv: Literal[True],
) -> TwoArrays:
    ...


@overload
def npexp(
    a: np.ndarray,
    deriv: Literal[False],
    bigx: float,
    verbose: bool,
) -> np.ndarray:
    ...


@overload
def npexp(
    a: np.ndarray,
    deriv: Literal[True],
    bigx: float,
    verbose: bool,
) -> TwoArrays:
    ...


@overload
def npexp(
    a: np.ndarray,
) -> np.ndarray:
    ...


def npexp(
    a: np.ndarray,
    deriv: bool = False,
    bigx: float = 30.0,
    verbose: bool = False,
) -> np.ndarray | TwoArrays:
    """
    $C^2$ extension of  $\exp(a)$ above `bigx`

    Args:
        a: a Numpy array
        deriv: if `True`, the first derivative is also returned
        bigx: an upper bound
        verbose: whether diagnoses are printed

    Returns:
        bigx: upper bound $\exp(a)$  $C^2$-extended above `bigx`,
        with its derivative if `deriv` is `True`
    """
    if np.max(a) < bigx:
        expa = np.exp(a)
        return (expa, expa) if deriv else expa
    else:
        exparr = np.exp(np.minimum(a, bigx))
        ebigx = exp(bigx)
        darr = a - bigx
        exparr_larger = ebigx * (1.0 + darr * (1.0 + 0.5 * darr))
        if verbose:
            n_large_args = np.sum(a > bigx)
            if n_large_args > 0:
                finals = "s" if n_large_args > 1 else ""
                print(
                    f"npexp: {n_large_args} argument{finals} larger than {bigx}: maxi = {np.max(a)}"
                )
        expa = np.where(a < bigx, exparr, exparr_larger)
        if deriv:
            der_exparr = np.exp(np.minimum(a, bigx))
            der_exparr_larger = ebigx * (1.0 + darr)
            der_expa = np.where(a < bigx, der_exparr, der_exparr_larger)
            return expa, der_expa
        else:
            return expa


@overload
def nppow(
    a: np.ndarray, b: int | float | np.ndarray, deriv: Literal[False]
) -> np.ndarray:
    ...


@overload
def nppow(
    a: np.ndarray, b: int | float | np.ndarray, deriv: Literal[True]
) -> ThreeArrays:
    ...


@overload
def nppow(
    a: np.ndarray,
    b: int | float | np.ndarray,
) -> np.ndarray:
    ...


def nppow(
    a: np.ndarray, b: int | float | np.ndarray, deriv: bool = False
) -> np.ndarray | ThreeArrays:
    """Evaluates $a^b$ element-by-element

    Args:
        a: a Numpy array
        b: if an array, it should have the same shape as `a`
        deriv: if `True`, the first derivatives wrt `a` and `b`
            are also returned

    Returns:
        an array of the same shape as `a`, and if `deriv` is `True`,
        the derivatives wrt `a` and `b`
    """
    mina = np.min(a)
    if mina <= 0:
        bs_error_abort("All elements of a must be positive!")

    if isinstance(b, (int, float)):
        a_pow_b = a**b
        if deriv:
            return (a**b, b * a_pow_b / a, a_pow_b * log(a))
        else:
            return a_pow_b
    else:
        if a.shape != b.shape:
            bs_error_abort(f"a has shape {a.shape} and b has shape {b.shape}")
        avec = a.ravel()
        bvec = b.ravel()
        a_pow_b = avec**bvec
        if deriv:
            der_wrt_a = a_pow_b * bvec / avec
            der_wrt_b = a_pow_b * nplog(avec)
            return (
                a_pow_b.reshape(a.shape),
                der_wrt_a.reshape(a.shape),
                der_wrt_b.reshape(a.shape),
            )
        else:
            return a_pow_b.reshape(a.shape)


def der_nppow(a: np.ndarray, b: int | float | np.ndarray) -> TwoArrays:
    """
    evaluates the derivatives in a and b of element-by-element $a^b$

    Args:
        a: input Numpy arrays
        b: a Numpy array of the same shape, or a scalar

    Returns:
        a pair of two arrays of the same shape as `a`
    """
    mina: float = np.min(a)
    if mina <= 0:
        print_stars("All elements of a must be positive in der_nppow!")
        sys.exit(1)

    if isinstance(b, (int, float)):
        a_pow_b = np.power(a, b)
        return (b * a_pow_b / a, a_pow_b * log(a))
    else:
        if a.shape != b.shape:
            print_stars(
                "nppow: b is not a number or an array of the same shape as a!"
            )
            sys.exit(1)
        avec = a.ravel()
        bvec = b.ravel()
        a_pow_b = avec**bvec
        der_wrt_a = a_pow_b * bvec / avec
        der_wrt_b = a_pow_b * nplog(avec)
        return der_wrt_a.reshape(a.shape), der_wrt_b.reshape(a.shape)


def check_gradient_scalar_function(
    fg: Any,
    p: np.ndarray,
    args: list[Any],
    mode: str = "central",
    EPS: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Checks the gradient of a scalar function.

    Args:
        fg: should return the scalar value, and the gradient if its `gr` argument is `True`
        p: where we are checking the gradient
        args: other arguments passed to `fg`
        mode: "central" or "forward" derivatives
        EPS: the step for forward or central derivatives

    Returns:
        the analytic and numeric gradients
    """
    # if not isinstance(fg, ScalarFunctionAndGradient):
    #     bs_error_abort("wrong type for fg")
    f0, f_grad = fg(p, args, True)

    print_stars("checking the gradient: analytic, numeric")

    g = np.zeros_like(p)
    sp: int = p.size
    if mode == "central":
        for i in range(sp):
            x = p[i]
            p1 = p.copy()
            p1[i] = x + EPS
            f_plus = fg(p1, args, False)
            p1[i] -= 2.0 * EPS
            f_minus = fg(p1, args, False)
            g[i] = (f_plus - f_minus) / (2.0 * EPS)
            print(f"{i}: {f_grad[i]}, {g[i]}")
    elif mode == "forward":
        for i in range(sp):
            x = p[i]
            p1 = p.copy()
            p1[i] = x + EPS
            f_plus = fg(p1, args, False)
            g[i] = (f_plus - f0) / EPS
            print(f"{i}: {f_grad[i]}, {g[i]}")
    else:
        bs_error_abort("mode must be 'central' or 'forward'")

    return f_grad, g
