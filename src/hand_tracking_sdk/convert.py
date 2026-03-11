"""Coordinate conversion utilities for HTS telemetry."""

from __future__ import annotations

from hand_tracking_sdk.frame import HandFrame
from hand_tracking_sdk.models import HandLandmarks, WristPose

# Type alias for a 3×3 matrix stored as a tuple-of-tuples.
Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

# Canonical basis transforms from Unity left-handed coordinates.
# Unity left-handed uses x=right, y=up, z=forward.
#
# RFU target basis uses x=right, y=forward, z=up.
# Mapping: (x, y, z) -> (x, z, y)
BASIS_UNITY_LEFT_TO_RFU: Matrix3x3 = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
)
#
# FLU target basis uses x=forward, y=left, z=up.
# Mapping: (x, y, z) -> (z, -x, y)
BASIS_UNITY_LEFT_TO_FLU: Matrix3x3 = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


def unity_left_to_right_position(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert a Unity left-handed position into a right-handed position.

    HTS documentation indicates flipping the Y axis for typical right-handed
    consumer stacks.

    :param x:
        Position X in Unity left-handed coordinates.
    :param y:
        Position Y in Unity left-handed coordinates.
    :param z:
        Position Z in Unity left-handed coordinates.
    :returns:
        Converted ``(x, y, z)`` in right-handed coordinates.
    """
    return (x, -y, z)


def unity_right_to_flu_position(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Unity right-handed coordinates into FLU basis.

    Unity right-handed coordinates use ``x=right, y=up, z=forward`` while
    common robotics FLU basis uses ``x=forward, y=left, z=up``.
    This function applies the required axis remapping and sign flips.
    See GitHub issue #2 for more details.

    :param x:
        Position X in Unity right-handed coordinates.
    :param y:
        Position Y in Unity right-handed coordinates.
    :param z:
        Position Z in Unity right-handed coordinates.
    :returns:
        Position converted into FLU basis.
    """
    return (z, -x, -y)


def unity_left_to_flu_position(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Unity left-handed coordinates directly into FLU basis.

    This composes :func:`unity_left_to_right_position` followed by
    :func:`unity_right_to_flu_position`, so Unity left-handed input is first
    converted to Unity right-handed coordinates, then to FLU.

    :param x:
        Position X in Unity left-handed coordinates.
    :param y:
        Position Y in Unity left-handed coordinates.
    :param z:
        Position Z in Unity left-handed coordinates.
    :returns:
        Position converted into FLU basis.
    """
    right_x, right_y, right_z = unity_left_to_right_position(x, y, z)
    return unity_right_to_flu_position(right_x, right_y, right_z)


def unity_left_to_right_quaternion(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float, float]:
    """Convert quaternion orientation from Unity left-handed to right-handed.

    The conversion applies basis transform ``R' = S * R * S`` with
    ``S = diag(1, -1, 1)``.

    :param qx:
        Quaternion X in Unity left-handed basis.
    :param qy:
        Quaternion Y in Unity left-handed basis.
    :param qz:
        Quaternion Z in Unity left-handed basis.
    :param qw:
        Quaternion W in Unity left-handed basis.
    :returns:
        Converted quaternion ``(qx, qy, qz, qw)`` in right-handed basis.
    """
    matrix = _quaternion_to_matrix(qx=qx, qy=qy, qz=qz, qw=qw)

    # Basis reflection matrix for Y-axis flip.
    s = (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    transformed = _matmul(_matmul(s, matrix), s)
    return _matrix_to_quaternion(transformed)


def convert_wrist_pose_unity_left_to_right(pose: WristPose) -> WristPose:
    """Convert one wrist pose from Unity left-handed to right-handed.

    :param pose:
        Wrist pose to convert.
    :returns:
        Converted wrist pose.
    """
    x, y, z = unity_left_to_right_position(pose.x, pose.y, pose.z)
    qx, qy, qz, qw = unity_left_to_right_quaternion(pose.qx, pose.qy, pose.qz, pose.qw)
    return WristPose(x=x, y=y, z=z, qx=qx, qy=qy, qz=qz, qw=qw)


def convert_landmarks_unity_left_to_right(landmarks: HandLandmarks) -> HandLandmarks:
    """Convert hand landmarks from Unity left-handed to right-handed.

    :param landmarks:
        Landmark set to convert.
    :returns:
        Converted landmark set preserving original point order.
    """
    converted = tuple(unity_left_to_right_position(x, y, z) for x, y, z in landmarks.points)
    return HandLandmarks(points=converted)


def convert_hand_frame_unity_left_to_right(frame: HandFrame) -> HandFrame:
    """Convert a full assembled hand frame from Unity left-handed to right-handed.

    Metadata fields are preserved and only geometry fields are transformed.

    :param frame:
        Input frame in Unity left-handed coordinates.
    :returns:
        Converted frame in right-handed coordinates.
    """
    return HandFrame(
        side=frame.side,
        frame_id=frame.frame_id,
        wrist=convert_wrist_pose_unity_left_to_right(frame.wrist),
        landmarks=convert_landmarks_unity_left_to_right(frame.landmarks),
        sequence_id=frame.sequence_id,
        recv_ts_ns=frame.recv_ts_ns,
        recv_time_unix_ns=frame.recv_time_unix_ns,
        source_ts_ns=frame.source_ts_ns,
        wrist_recv_ts_ns=frame.wrist_recv_ts_ns,
        landmarks_recv_ts_ns=frame.landmarks_recv_ts_ns,
        source_frame_seq=frame.source_frame_seq,
    )


def _quaternion_to_matrix(
    *,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Convert quaternion components into a 3x3 rotation matrix.

    :param qx:
        Quaternion X.
    :param qy:
        Quaternion Y.
    :param qz:
        Quaternion Z.
    :param qw:
        Quaternion W.
    :returns:
        Rotation matrix corresponding to the input quaternion.
    """
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _matrix_to_quaternion(
    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix into quaternion components.

    :param matrix:
        Rotation matrix.
    :returns:
        Normalized quaternion ``(qx, qy, qz, qw)``.
    """
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    return _normalize_quaternion(qx=qx, qy=qy, qz=qz, qw=qw)


def _normalize_quaternion(
    *,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float, float]:
    """Normalize quaternion values and fall back to identity on zero norm.

    :param qx:
        Quaternion X.
    :param qy:
        Quaternion Y.
    :param qz:
        Quaternion Z.
    :param qw:
        Quaternion W.
    :returns:
        Normalized quaternion tuple.
    """
    norm = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def _matmul(
    a: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    b: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Multiply two 3x3 matrices.

    :param a:
        Left-hand matrix operand.
    :param b:
        Right-hand matrix operand.
    :returns:
        Matrix product ``a * b``.
    """
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def _transpose(m: Matrix3x3) -> Matrix3x3:
    """Transpose a 3×3 matrix.

    :param m:
        Input matrix.
    :returns:
        Transposed matrix.
    """
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


# ---------------------------------------------------------------------------
# General-purpose basis transform utilities
# ---------------------------------------------------------------------------


def basis_transform_position(
    pos: tuple[float, float, float],
    basis: Matrix3x3,
) -> tuple[float, float, float]:
    """Apply a basis matrix to a position vector.

    Computes ``result = basis @ pos``.

    :param pos:
        Input position ``(x, y, z)``.
    :param basis:
        3×3 basis change matrix.
    :returns:
        Transformed position.
    """
    return (
        basis[0][0] * pos[0] + basis[0][1] * pos[1] + basis[0][2] * pos[2],
        basis[1][0] * pos[0] + basis[1][1] * pos[1] + basis[1][2] * pos[2],
        basis[2][0] * pos[0] + basis[2][1] * pos[1] + basis[2][2] * pos[2],
    )


def basis_transform_rotation(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    basis: Matrix3x3,
) -> tuple[float, float, float, float]:
    """Transform a quaternion orientation via a basis change matrix.

    Computes ``R_new = basis @ R @ basis^T`` and returns the result as a
    normalized quaternion ``(qx, qy, qz, qw)``.

    :param qx:
        Quaternion X component.
    :param qy:
        Quaternion Y component.
    :param qz:
        Quaternion Z component.
    :param qw:
        Quaternion W component.
    :param basis:
        3×3 basis change matrix.
    :returns:
        Transformed quaternion ``(qx, qy, qz, qw)``.
    """
    rot = _quaternion_to_matrix(qx=qx, qy=qy, qz=qz, qw=qw)
    transformed = _matmul(_matmul(basis, rot), _transpose(basis))
    return _matrix_to_quaternion(transformed)


def basis_transform_rotation_matrix(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    basis: Matrix3x3,
) -> Matrix3x3:
    """Transform a quaternion orientation via a basis change matrix.

    Same operation as :func:`basis_transform_rotation` but returns the
    result as a 3×3 rotation matrix instead of a quaternion.

    :param qx:
        Quaternion X component.
    :param qy:
        Quaternion Y component.
    :param qz:
        Quaternion Z component.
    :param qw:
        Quaternion W component.
    :param basis:
        3×3 basis change matrix.
    :returns:
        Transformed 3×3 rotation matrix.
    """
    rot = _quaternion_to_matrix(qx=qx, qy=qy, qz=qz, qw=qw)
    return _matmul(_matmul(basis, rot), _transpose(basis))


def unity_left_to_rfu_position(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Unity left-handed coordinates into RFU basis."""
    return basis_transform_position((x, y, z), BASIS_UNITY_LEFT_TO_RFU)


def unity_left_to_rfu_rotation(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float, float]:
    """Convert Unity left-handed quaternion into RFU basis."""
    return basis_transform_rotation(qx, qy, qz, qw, BASIS_UNITY_LEFT_TO_RFU)


def unity_left_to_rfu_rotation_matrix(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> Matrix3x3:
    """Convert Unity left-handed quaternion into RFU rotation matrix."""
    return basis_transform_rotation_matrix(qx, qy, qz, qw, BASIS_UNITY_LEFT_TO_RFU)


def unity_left_to_flu_rotation(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float, float]:
    """Convert Unity left-handed quaternion into FLU basis."""
    return basis_transform_rotation(qx, qy, qz, qw, BASIS_UNITY_LEFT_TO_FLU)


def unity_left_to_flu_rotation_matrix(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> Matrix3x3:
    """Convert Unity left-handed quaternion into FLU rotation matrix."""
    return basis_transform_rotation_matrix(qx, qy, qz, qw, BASIS_UNITY_LEFT_TO_FLU)
