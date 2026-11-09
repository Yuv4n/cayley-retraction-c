# Why the update preserves orthogonality

Let K = (η/2)A, so that W⁺ = QW with Q = (I + K)⁻¹(I − K).

If Q is orthogonal and W is orthogonal, then W⁺ᵀW⁺ = WᵀQᵀQW = WᵀW = I. Everything therefore rests on Q, and the argument has three parts.

## 1. K is skew-symmetric

Aᵀ = (GWᵀ − WGᵀ)ᵀ = WGᵀ − GWᵀ = −A, and scaling by η/2 does not change that, so Kᵀ = −K. Nothing about W or G is used here. The same holds for any matrices G and W.

## 2. I + K and I − K are invertible

For skew-symmetric K and any real vector x, the scalar xᵀKx equals its own transpose, which is xᵀKᵀx = −xᵀKx. It is therefore zero.

Suppose (I + K)x = 0. Then 0 = xᵀ(I + K)x = xᵀx + xᵀKx = xᵀx, so x = 0, and I + K has a trivial null space. The same argument applies to I − K, because −K is also skew-symmetric. Both matrices are invertible for every real η, so the update is always defined.

## 3. Q is orthogonal

From Kᵀ = −K we get (I + K)ᵀ = I − K and (I − K)ᵀ = I + K. So

Qᵀ = (I − K)ᵀ((I + K)⁻¹)ᵀ = (I + K)(I − K)⁻¹,

and

QᵀQ = (I + K)(I − K)⁻¹(I + K)⁻¹(I − K).

I + K and I − K are both polynomials in K, so they commute, and so do their inverses. Swapping the two middle factors gives

QᵀQ = (I + K)(I + K)⁻¹(I − K)⁻¹(I − K) = I.

Hence Q is orthogonal, and so is W⁺.

## Determinant

det(I − K) = det((I + K)ᵀ) = det(I + K), so det Q = 1 and det W⁺ = det W. The update never turns a rotation into a reflection or the reverse.

## Limits of the argument

All of this is exact arithmetic. In floating point the elimination and the matrix products each add rounding error, so W⁺ is orthogonal only to within a small residual, and that residual can grow over many steps. [validation.md](validation.md) describes how it will be measured.

## Why it is a retraction

Since I − K = 2I − (I + K), the update can be written W⁺ = W − η(I + K)⁻¹AW. For small η this is W − ηAW to first order, which is a step in a direction built from the gradient. The Cayley form bends that step onto the set of orthogonal matrices, which is what the term retraction means.
