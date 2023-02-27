import numpy as np
import typer

from .choo_siow import entropy_choo_siow, entropy_choo_siow_numeric
from .choo_siow_gender_heteroskedastic import (
    entropy_choo_siow_gender_heteroskedastic,
    entropy_choo_siow_gender_heteroskedastic_numeric,
)
from .choo_siow_heteroskedastic import (
    entropy_choo_siow_heteroskedastic,
    entropy_choo_siow_heteroskedastic_numeric,
)
from .min_distance import estimate_semilinear_mde
from .model_classes import ChooSiowPrimitives, NestedLogitPrimitives
from .nested_logit import setup_standard_nested_logit
from .poisson_glm import choo_siow_poisson_glm
from .utils import nprepeat_col, nprepeat_row, print_stars

app = typer.Typer()

seed = 177564


def create_choo_siow_homo(n_households: int, seed: int):
    """Create a set of observations from a Choo Siow homoskedastic market.

    Args:
        n_households: number of observations
        seed: for RNG

    Returns:
        phi_bases: the basis functions
        lambda_true: their coefficients
        mus_sim: the simulated Matching
    """
    rng = np.random.default_rng(seed=seed)
    X, Y, K = 20, 20, 8
    # set of 8 basis functions
    phi_bases = np.zeros((X, Y, K))
    phi_bases[:, :, 0] = 1.0
    vec_x = np.arange(X)
    vec_y = np.arange(Y)
    phi_bases[:, :, 1] = nprepeat_col(vec_x, Y)
    phi_bases[:, :, 2] = nprepeat_row(vec_y, X)
    phi_bases[:, :, 3] = phi_bases[:, :, 1] * phi_bases[:, :, 1]
    phi_bases[:, :, 4] = phi_bases[:, :, 1] * phi_bases[:, :, 2]
    phi_bases[:, :, 5] = phi_bases[:, :, 2] * phi_bases[:, :, 2]
    for i in range(X):
        for j in range(i, Y):
            phi_bases[i, j, 6] = 1
            phi_bases[i, j, 7] = i - j
    lambda_true = np.array([1.0, 0.0, 0.0, -0.01, 0.02, -0.01, 0.5, 0.0])

    # we simulate a Choo and Siow population
    #  with equal numbers of men and women of each type
    lambda_true = rng.normal(size=K)
    phi_bases = rng.normal(size=(X, Y, K))
    n = np.ones(X)
    m = np.ones(Y)
    Phi = phi_bases @ lambda_true
    choo_siow_instance = ChooSiowPrimitives(Phi, n, m)
    mus_sim = choo_siow_instance.simulate(n_households, seed=seed)
    choo_siow_instance.describe()
    return phi_bases, lambda_true, mus_sim


@app.command()
def try_mde_homo(n_households: int, numeric: bool = False) -> None:
    """
    Try an MDE estimate of Choo and Siow homoskedastic.

    Args:
        n_households: the number of observations
        numeric: if True we use numerical Hessians
    """
    str_num = "numeric" if numeric else "analytic"
    typer.echo(
        f"""
    Trying MDE on Choo and Siow homoskedastic
    with {str_num} Hessians
    """
    )
    phi_bases, lambda_true, mus_sim = create_choo_siow_homo(
        n_households, seed
    )

    entropy_model = (
        entropy_choo_siow_numeric if numeric else entropy_choo_siow
    )
    true_coeffs = lambda_true

    print_stars(entropy_model.description)

    mde_results = estimate_semilinear_mde(mus_sim, phi_bases, entropy_model)
    mde_results.print_results(true_coeffs=true_coeffs)


@app.command()
def try_mde_gender_hetero(n_households: int, numeric: bool = False) -> None:
    """
    Try an MDE estimate of Choo and Siow gender heteroskedastic.

    Args:
        n_households: the number of observations
        numeric: if True we use numerical Hessians
    """
    str_num = "numeric" if numeric else "analytic"
    typer.echo(
        f"""
    Trying MDE on Choo and Siow gender heteroskedastic
    with {str_num} Hessians
    """
    )
    phi_bases, lambda_true, mus_sim = create_choo_siow_homo(
        n_households, seed
    )

    entropy_model = (
        entropy_choo_siow_gender_heteroskedastic_numeric
        if numeric
        else entropy_choo_siow_gender_heteroskedastic
    )
    true_coeffs = np.concatenate((np.ones(1), lambda_true))

    print_stars(entropy_model.description)

    mde_results = estimate_semilinear_mde(mus_sim, phi_bases, entropy_model)
    mde_results.print_results(true_coeffs=true_coeffs, n_alpha=1)


@app.command()
def try_mde_hetero(n_households: int, numeric: bool = False) -> None:
    """
    Try an MDE estimate of Choo and Siow heteroskedastic.

    Args:
        n_households: the number of observations
        numeric: if True we use numerical Hessians
    """
    str_num = "numeric" if numeric else "analytic"
    typer.echo(
        f"""
    Trying MDE on Choo and Siow heteroskedastic
    with {str_num} Hessians
    """
    )
    phi_bases, lambda_true, mus_sim = create_choo_siow_homo(
        n_households, seed
    )

    entropy_model = (
        entropy_choo_siow_heteroskedastic_numeric
        if numeric
        else entropy_choo_siow_heteroskedastic
    )
    X, Y = phi_bases.shape[:-1]
    true_coeffs = np.concatenate((np.ones(X + Y - 1), lambda_true))

    print_stars(entropy_model.description)

    mde_results = estimate_semilinear_mde(mus_sim, phi_bases, entropy_model)
    mde_results.print_results(true_coeffs=true_coeffs, n_alpha=X + Y - 1)


@app.command()
def try_mde_nested_logit(n_households: int, numeric: bool = False) -> None:
    """
    Try an MDE estimate of a nested logit.

    Args:
        n_households: the number of observations
        numeric: if True we use numerical Hessians
    """
    str_num = "numeric" if numeric else "analytic"
    typer.echo(
        f"""
    Trying MDE on a nested logit
    with {str_num} Hessians
    """
    )

    phi_bases, true_betas, _ = create_choo_siow_homo(n_households, seed)
    X, Y = phi_bases.shape[:-1]

    # Nests and nest parameters for our two-level nested logit
    #  0 is the first nest;
    #   all other nests and nest parameters are type-independent
    # each x has the same nests over 1, ..., Y
    nests_for_each_y = [
        list(range(1, Y // 2 + 1)),
        list(range(Y // 2 + 1, Y + 1)),
    ]
    # each y has the same nests over 1, ..., X
    nests_for_each_x = [
        list(range(1, X // 2 + 1)),
        list(range(X // 2 + 1, X + 1)),
    ]

    (
        entropy_nested_logit,
        entropy_nested_logit_numeric,
    ) = setup_standard_nested_logit(nests_for_each_x, nests_for_each_y)

    n_rhos, n_deltas = len(nests_for_each_x), len(nests_for_each_y)
    n_alphas = n_rhos + n_deltas

    true_alphas = np.full(n_alphas, 0.5)

    true_coeffs = np.concatenate((true_alphas, true_betas))

    Phi = phi_bases @ true_betas
    n = np.ones(X)
    m = np.ones(Y)

    nested_logit_instance = NestedLogitPrimitives(
        Phi, n, m, nests_for_each_x, nests_for_each_y, true_alphas
    )

    entropy_model = (
        entropy_nested_logit_numeric if numeric else entropy_nested_logit
    )

    mus_sim = nested_logit_instance.simulate(n_households, seed)

    print_stars(entropy_model.description)
    mde_results = estimate_semilinear_mde(
        mus_sim,
        phi_bases,
        entropy_model,
        additional_parameters=entropy_model.additional_parameters,
    )

    mde_results.print_results(true_coeffs=true_coeffs, n_alpha=n_alphas)


@app.command()
def try_poisson(n_households: int):
    """
    Try the Poisson method
    """
    typer.echo("Trying Poisson")

    phi_bases, lambda_true, mus_sim = create_choo_siow_homo(
        n_households, seed
    )

    _, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()

    results = choo_siow_poisson_glm(mus_sim, phi_bases)

    # compare true and estimated parameters
    results.print_results(
        lambda_true,
        u_true=-np.log(mux0_sim / n_sim),
        v_true=-np.log(mu0y_sim / m_sim),
    )
