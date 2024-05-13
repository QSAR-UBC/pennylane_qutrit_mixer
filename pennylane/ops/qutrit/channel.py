# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-arguments
"""
This module contains the available built-in noisy qutrit
quantum channels supported by PennyLane, as well as their conventions.
"""
import numpy as np
from pennylane import math
from pennylane.operation import Channel

QUDIT_DIM = 3


class QutritDepolarizingChannel(Channel):
    r"""
    Single-qutrit symmetrically depolarizing error channel.
    This channel is modelled by the Kraus matrices generated by the following relationship:

    .. math::
        X = \begin{bmatrix}
                0 & 1 & 0 \\
                0 & 0 & 1 \\
                1 & 0 & 0
                \end{bmatrix},\;
        Z = \begin{bmatrix}
                1 & 0 & 0\\
                0 & \omega & 0\\
                0 & 0 & \omega^2
                \end{bmatrix}

    .. math::
        K_0 = \sqrt{1-p} \begin{bmatrix}
                1 & 0 & 0\\
                0 & 1 & 0\\
                0 & 0 & 1
                \end{bmatrix}

    .. math::
        K_{i,j} = \sqrt{\frac{p}{8}}X^iZ^j, \{i,j\} \neq \{0,0\}

    These relations create the following Kraus matrices:

    .. math::
        \begin{matrix}
            K_0 = \sqrt{1-p} \begin{bmatrix}
                1 & 0 & 0\\
                0 & 1 & 0\\
                0 & 0 & 1
                \end{bmatrix}&
            K_1 = \sqrt{\frac{p}{8}}\begin{bmatrix}
                1 & 0 & 0\\
                0 & \omega & 0\\
                0 & 0 & \omega^2
                \end{bmatrix}&
            K_2 = \sqrt{\frac{p}{8}}\begin{bmatrix}
                1 & 0 & 0\\
                0 & \omega^2 & 0\\
                0 & 0 & \omega^4
                \end{bmatrix}\\
            K_3 = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & 1 & 0 \\
                0 & 0 & 1 \\
                1 & 0 & 0
                \end{bmatrix}&
            K_4 = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & \omega & 0 \\
                0 & 0 & \omega^2 \\
                1 & 0 & 0
                \end{bmatrix}&
            K_5 = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & \omega^2 & 0 \\
                0 & 0 & \omega \\
                1 & 0 & 0
                \end{bmatrix}\\
            K_6 = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & 0 & 1 \\
                1 & 0 & 0 \\
                0 & 1 & 0
                \end{bmatrix}&
            K_7 = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & 0 & \omega^2 \\
                1 & 0 & 0 \\
                0 & \omega & 0
                \end{bmatrix}&
            K_8 = \sqrt{\frac{p}{8}}\begin{bmatrix}
                0 & 0 & \omega \\
                1 & 0 & 0 \\
                0 & \omega^2 & 0
                \end{bmatrix}
        \end{matrix}


    Where :math:`\omega=\exp(\frac{2\pi}{3})`  is the third root of unity.
    where :math:`p \in [0, 1]` is the depolarization probability and is equally
    divided in the application of all qutrit Pauli operators.

    .. note::

        The Kraus operators :math:`\{K_0 \ldots K_8\}` used are the representations of the single qutrit Pauli group.
        These Pauli group operators are defined in [`1 <https://arxiv.org/pdf/quant-ph/9802007>`_] (Eq. 5).
        The Kraus Matrices we use are adapted from [`2 <https://doi.org/10.48550/arXiv.1905.10481>`_] (Eq. 5).
        For this definition, please make a note of the following:

        * For :math:`p = 0`, the channel will be an Identity channel, i.e., a noise-free channel.
        * For :math:`p = \frac{8}{9}`, the channel will be a fully depolarizing channel.
        * For :math:`p = 1`, the channel will be a uniform error channel.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1

    Args:
        p (float): Each qutrit Pauli operator is applied with probability :math:`\frac{p}{8}`
        wires (Sequence[int] or int): the wire the channel acts on
        id (str or None): String representing the operation (optional)
    """

    num_params = 1
    num_wires = 1
    grad_method = "A"
    grad_recipe = ([[1, 0, 1], [-1, 0, 0]],)

    def __init__(self, p, wires, id=None):
        super().__init__(p, wires=wires, id=id)

    @staticmethod
    def compute_kraus_matrices(p):  # pylint:disable=arguments-differ
        r"""Kraus matrices representing the qutrit depolarizing channel.

        Args:
            p (float): each qutrit Pauli gate is applied with probability :math:`\frac{p}{8}`

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> np.round(qml.QutritDepolarizingChannel.compute_kraus_matrices(0.5), 3)
        array([[[ 0.707+0.j   ,  0.   +0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.707+0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.   +0.j   ,  0.707+0.j   ]],

          [[ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   , -0.125+0.217j,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.   +0.j   , -0.125-0.217j]],

          [[ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   , -0.125-0.217j,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.   +0.j   , -0.125+0.217j]],

          [[ 0.   +0.j   ,  0.25 +0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.   +0.j   ,  0.25 +0.j   ],
           [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ]],

          [[ 0.   +0.j   , -0.125+0.217j,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.   +0.j   , -0.125-0.217j],
           [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ]],

          [[ 0.   +0.j   , -0.125-0.217j,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.   +0.j   , -0.125+0.217j],
           [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ]],

          [[ 0.   +0.j   ,  0.   +0.j   ,  0.25 +0.j   ],
           [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   ,  0.25 +0.j   ,  0.   +0.j   ]],

          [[ 0.   +0.j   ,  0.   +0.j   , -0.125-0.217j],
           [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   , -0.125+0.217j,  0.   +0.j   ]],

          [[ 0.   +0.j   ,  0.   +0.j   , -0.125+0.217j],
           [ 0.25 +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
           [ 0.   +0.j   , -0.125-0.217j,  0.   +0.j   ]]])
        """
        if not math.is_abstract(p) and not 0.0 <= p <= 1.0:
            raise ValueError("p must be in the interval [0,1]")

        interface = math.get_interface(p)

        w = math.exp(2j * np.pi / 3)
        one = 1
        z = 0

        if interface == "tensorflow":
            p = math.cast_like(p, 1j)
            w = math.cast_like(w, p)
            one = math.cast_like(one, p)
            z = math.cast_like(z, p)

        w2 = w**2

        depolarizing_mats = [
            [[one, z, z], [z, w, z], [z, z, w2]],
            [[one, z, z], [z, w2, z], [z, z, w]],
            [[z, one, z], [z, z, one], [one, z, z]],
            [[z, w, z], [z, z, w2], [one, z, z]],
            [[z, w2, z], [z, z, w], [one, z, z]],
            [[z, z, one], [one, z, z], [z, one, z]],
            [[z, z, w2], [one, z, z], [z, w, z]],
            [[z, z, w], [one, z, z], [z, w2, z]],
        ]
        normalization = math.sqrt(p / 8 + math.eps)
        Ks = [normalization * math.array(m, like=interface) for m in depolarizing_mats]
        identity = math.sqrt(1 - p + math.eps) * math.array(
            math.eye(QUDIT_DIM, dtype=complex), like=interface
        )

        return [identity] + Ks
