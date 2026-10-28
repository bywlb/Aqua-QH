
def loss_and_gradient(theta: np.ndarray, records: list[dict],
                      lambda_reg: float = 1e-4) -> tuple[float, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    grad = lambda_reg * theta.copy()
    loss = 0.5 * lambda_reg * float(np.dot(theta, theta))
    if not records:
        return loss, grad
    u = np.array([r["u_idx"] for r in records], dtype=np.int64)
    v = np.array([r["v_idx"] for r in records], dtype=np.int64)
    p = np.array([r["prob_u_win"] for r in records], dtype=float)
    delta = theta[u] - theta[v]
    # Stable binary soft-label cross entropy.
    loss += float(np.sum(p * np.logaddexp(0.0, -delta) + (1.0-p) * np.logaddexp(0.0, delta)))
    diff = expit(delta) - p
    np.add.at(grad, u, diff)
    np.add.at(grad, v, -diff)
    return loss, grad


def aggregate_soft_bt(n_docs: int, records: list[dict],
                      lambda_reg: float = 1e-4, max_iter: int = 500) -> np.ndarray:
    if n_docs < 1:
        raise ValueError("n_docs must be at least one")
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be nonnegative")
    if not records:
        return np.zeros(n_docs)
    for record in records:
        if not 0 <= record["u_idx"] < n_docs or not 0 <= record["v_idx"] < n_docs:
            raise ValueError("Pair indices out of range")
        if not 0 <= record["prob_u_win"] <= 1:
            raise ValueError("Invalid probability")

    result = minimize(loss_and_gradient, np.zeros(n_docs),
                      args=(records, lambda_reg), method="L-BFGS-B", jac=True,
                      options={"maxiter": max_iter, "ftol": 1e-10})
    if not result.success:
        import warnings
        warnings.warn(f"BT solver stopped: {result.message}", RuntimeWarning)
    theta = np.asarray(result.x)
    return theta - np.mean(theta)
