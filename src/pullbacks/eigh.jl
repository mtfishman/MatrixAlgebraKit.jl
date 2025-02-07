function eigh_full_pullback!(ΔA::AbstractMatrix, DV, ΔDV;
                             tol::Real=default_pullback_gaugetol(DV[1]))

    # Basic size checks and determination
    Dmat, V = DV
    D = diagview(Dmat)
    ΔDmat, ΔV = ΔDV
    n = LinearAlgebra.checksquare(V)
    n == length(D) || throw(DimensionMismatch())

    if !iszerotangent(ΔV)
        VdΔV = V' * ΔV
        aVdΔV = rmul!(VdΔV - VdΔV', 1 / 2)

        mask = abs.(D' .- D) .< tol
        Δgauge = norm(view(aVdΔV, mask))
        Δgauge < tol ||
            @warn "`eigh` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

        aVdΔV .*= safe_inv.(D' .- D, tol)

        if !iszerotangent(ΔDmat)
            diagview(aVdΔV, diagind(aVdΔV)) .+= real.(diagview(ΔDmat))
        end
        # recylce VdΔV space
        ΔA = mul!(ΔA, mul!(VdΔV, V, aVdΔV), V', 1, 1)
    elseif !iszerotangent(ΔDmat)
        ΔA = mul!(ΔA, V * Diagonal(real(diagview(ΔDmat))), V', 1, 1)
    end
    return ΔA
end
