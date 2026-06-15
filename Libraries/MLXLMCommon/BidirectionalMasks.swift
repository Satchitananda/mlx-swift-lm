// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Bidirectional attention mask: every query position attends to every kv
/// position equally (no causal restriction).
///
/// Used by the MTP drafter, whose queries sit at a single constant position
/// outside the target's KV cache and need full visibility into the target's
/// shared K/V pool. Returned as an additive mask (`0` for attend, `-inf` for
/// mask) compatible with `MLXFast.ScaledDotProductAttentionMaskMode.array`.
///
/// - Parameters:
///   - queryLen: number of query tokens (typically `1` for MTP drafting)
///   - kvLen: total kv positions in the shared pool
///   - dtype: array dtype (must match the queries' dtype)
/// - Returns: `[queryLen, kvLen]` array of zeros.
public func createBidirectionalMask(
    queryLen: Int,
    kvLen: Int,
    dtype: DType
) -> MLXArray {
    MLXArray.zeros([queryLen, kvLen], dtype: dtype)
}

/// Bidirectional sliding-window attention mask: each query attends to the
/// most-recent `windowSize` kv positions and is blocked from all older ones.
/// When `windowSize >= kvLen` (all positions fit inside the window), the
/// helper early-exits to an all-zeros mask.
///
/// The non-degenerate path (`kvLen > windowSize`) attends to indices
/// `[kvLen - windowSize, kvLen)` — the LAST `windowSize` positions — and
/// masks the remainder with `-inf`. This is correct when the shared KV
/// snapshot comes from a `RotatingKVCache` whose oldest entry is at index 0
/// and the newest entry is at index `kvLen - 1`.
///
/// - Parameters:
///   - queryLen: number of query tokens
///   - kvLen: total kv positions in the shared pool
///   - windowSize: sliding window size
///   - dtype: array dtype (must match the queries' dtype)
/// - Returns: `[queryLen, kvLen]` additive mask.
public func createBidirectionalSlidingWindowMask(
    queryLen: Int,
    kvLen: Int,
    windowSize: Int,
    dtype: DType
) -> MLXArray {
    if windowSize >= kvLen {
        return MLXArray.zeros([queryLen, kvLen], dtype: dtype)
    }
    // Attend to the newest `windowSize` KV entries (indices [kvLen-windowSize, kvLen)).
    let kIdx = MLXArray(Int32(0) ..< Int32(kvLen))
    let attend = kIdx .>= Int32(kvLen - windowSize)
    let row = MLX.where(
        attend,
        MLXArray(0, dtype: dtype),
        MLXArray(-Float.infinity, dtype: dtype)
    )
    return broadcast(row[.newAxis, 0...], to: [queryLen, kvLen])
}
