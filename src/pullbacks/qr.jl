# TODO: test wether this works equally well for `qr_compact` and `qr_full`
function qr_compact_pullback!(ΔA::AbstractMatrix, QR, ΔQR;
                              tol::Real=default_pullback_gaugetol(QR[2]))
    Q, R = QR
    ΔQ, ΔR = ΔQR

    Rd = diagview(R)
    p = findlast(>=(tol) ∘ abs, Rd)
    m, n = size(R)

    Q1 = view(Q, :, 1:p)
    R1 = view(R, 1:p, :)
    R11 = view(R, 1:p, 1:p)

    ΔA1 = view(ΔA, :, 1:p)

    ΔQ1 = view(ΔQ, :, 1:p)

    M = zero!(similar(R, (p, p)))
    if !iszerotangent(ΔR)
        ΔR1 = view(ΔR, 1:p, :)
        M = mul!(M, ΔR1, R1', 1, 1)
    end
    if !iszerotangent(ΔQ)
        ΔQ1 = view(ΔQ, :, 1:p)
        M = mul!(M, Q1', ΔQ1, -1, 1)
        ΔA1 .+= ΔQ1
    end
    view(M, lowertriangularind(M)) .= conj.(view(M, uppertriangularind(M)))
    if eltype(M) <: Complex
        Md = diagview(M)
        Md .= real.(Md)
    end
    ΔA1 = mul!(ΔA1, Q1, M, +1, 1)

    if n > p && !iszerotangent(ΔR)
        R12 = view(R, 1:p, (p + 1):n)
        ΔA2 = view(ΔA, :, (p + 1):n)
        ΔR12 = view(ΔR, 1:p, (p + 1):n)

        ΔA2 = mul!(ΔA2, Q1, ΔR12, 1, 1)
        ΔA1 = mul!(ΔA1, ΔA2, R12', -1, 1)
    end
    if m > p && !iszerotangent(ΔQ) # case where R is not full rank
        Q2 = view(Q, :, (p + 1):m)
        ΔQ2 = view(ΔQ, :, (p + 1):m)
        Q1dΔQ2 = Q1' * ΔQ2
        Δgauge = norm(mul!(copy(ΔQ2), Q1, Q1dΔQ2, -1, 1), Inf)
        Δgauge < tol ||
            @warn "`qr` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"
        ΔA1 = mul!(ΔA1, Q2, Q1dΔQ2', -1, 1)
    end
    rdiv!(ΔA1, UpperTriangular(R11)')
    return ΔA
end
