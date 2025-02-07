"""
    svd_compact_pullback!(ΔA, USVᴴ, ΔUSVᴴ; tol::Real=default_pullback_gaugetol(S))

Adds the pullback from the SVD of `A` to `ΔA` given the output USVᴴ of `svd_compact`
or `svd_full` and the cotangent `ΔUSVᴴ`. Here, `ΔUSVᴴ` can correspond to the cotangent
an `svd_trunc` call, where `ΔU`, `ΔS` and `ΔVᴴ` have size `(m, p)`, `(p, p)` and `(p, n)`
respectively.
"""
function svd_compact_pullback!(ΔA::AbstractMatrix, USVᴴ, ΔUSVᴴ;
                               tol::Real=default_pullback_gaugetol(USVᴴ[2]))
    U, Smat, Vᴴ = USVᴴ
    S = diagview(Smat)
    ΔU, ΔSmat, ΔVᴴ = ΔUSVᴴ
    # Basic size checks and determination
    m, n = size(U, 1), size(Vᴴ, 2)
    size(U, 2) == size(Vᴴ, 1) == length(S) == min(m, n) || throw(DimensionMismatch())
    p = -1
    if !iszerotangent(ΔU)
        m == size(ΔU, 1) || throw(DimensionMismatch())
        p = size(ΔU, 2)
    end
    if !iszerotangent(ΔVᴴ)
        n == size(ΔVᴴ, 2) || throw(DimensionMismatch())
        if p == -1
            p = size(ΔVᴴ, 1)
        else
            p == size(ΔVᴴ, 1) || throw(DimensionMismatch())
        end
    end
    if !iszerotangent(ΔSmat)
        ΔS = diagview(ΔSmat)
        if p == -1
            p = length(ΔS)
        else
            p == length(ΔS) || throw(DimensionMismatch())
        end
    end
    Up = view(U, :, 1:p)
    Vp = view(Vᴴ, 1:p, :)'
    Sp = view(S, 1:p)

    # rank
    r = findlast(>=(tol), S)

    # compute antihermitian part of projection of ΔU and ΔV onto U and V
    # also already subtract this projection from ΔU and ΔV
    if !iszerotangent(ΔU)
        UΔU = Up' * ΔU
        aUΔU = rmul!(UΔU - UΔU', 1 / 2)
        if m > p
            ΔU -= Up * UΔU
        end
    else
        aUΔU = fill!(similar(U, (p, p)), 0)
    end
    if !iszerotangent(ΔVᴴ)
        VΔV = Vp' * ΔVᴴ'
        aVΔV = rmul!(VΔV - VΔV', 1 / 2)
        if n > p
            ΔVᴴ -= VΔV' * Vp'
        end
    else
        aVΔV = fill!(similar(Vᴴ, (p, p)), 0)
    end

    # check whether cotangents arise from gauge-invariance objective function
    mask = abs.(Sp' .- Sp) .< tol
    Δgauge = norm(view(aUΔU, mask) + view(aVΔV, mask), Inf)
    if p > r
        rprange = (r + 1):p
        Δgauge = max(Δgauge, norm(view(aUΔU, rprange, rprange), Inf))
        Δgauge = max(Δgauge, norm(view(aVΔV, rprange, rprange), Inf))
    end
    Δgauge < tol ||
        @warn "`svd` cotangents sensitive to gauge choice: (|Δgauge| = $Δgauge)"

    UdΔAV = (aUΔU .+ aVΔV) .* safe_inv.(Sp' .- Sp, tol) .+
            (aUΔU .- aVΔV) .* safe_inv.(Sp' .+ Sp, tol)
    if !iszerotangent(ΔSmat)
        ΔS = diagview(ΔSmat)
        diagview(UdΔAV) .+= real.(ΔS)
    end
    ΔA = mul!(ΔA, Up, UdΔAV * Vp', 1, 1) # add the contribution to ΔA

    if r > p # contribution from truncation
        Ur = view(U, :, (p + 1):r)
        Vr = view(Vᴴ, (p + 1):r, :)'
        Sr = view(S, (p + 1):r)

        if !iszerotangent(ΔU)
            UrΔU = Ur' * ΔU
            if m > r
                ΔU -= Ur * UrΔU # subtract this part from ΔU
            end
        else
            UrΔU = fill!(similar(U, (r - p, p)), 0)
        end
        if !iszerotangent(ΔVᴴ)
            VrΔV = Vr' * ΔVᴴ'
            if n > r
                ΔVᴴ -= VrΔV' * Vr' # subtract this part from ΔV
            end
        else
            VrΔV = fill!(similar(Vᴴ, (r - p, p)), 0)
        end

        X = (1 // 2) .* ((UrΔU .+ VrΔV) .* safe_inv.(Sp' .- Sr, tol) .+
                         (UrΔU .- VrΔV) .* safe_inv.(Sp' .+ Sr, tol))
        Y = (1 // 2) .* ((UrΔU .+ VrΔV) .* safe_inv.(Sp' .- Sr, tol) .-
                         (UrΔU .- VrΔV) .* safe_inv.(Sp' .+ Sr, tol))

        # ΔA += Ur * X * Vp' + Up * Y' * Vr'
        ΔA = mul!(ΔA, Ur, X * Vp', 1, 1)
        ΔA = mul!(ΔA, Up * Y', Vr', 1, 1)
    end

    if m > max(r, p) && !iszerotangent(ΔU) # remaining ΔU is already orthogonal to U[:,1:max(p,r)]
        # ΔA += (ΔU .* safe_inv.(Sp', tol)) * Vp'
        ΔA = mul!(ΔA, ΔU .* safe_inv.(Sp', tol), Vp', 1, 1)
    end
    if n > max(r, p) && !iszerotangent(ΔVᴴ) # remaining ΔV is already orthogonal to V[:,1:max(p,r)]
        # ΔA += U * (safe_inv.(Sp, tol) .* ΔVᴴ)
        ΔA = mul!(ΔA, Up, safe_inv.(Sp, tol) .* ΔVᴴ, 1, 1)
    end
    return ΔA
end
