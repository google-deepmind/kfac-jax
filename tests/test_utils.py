# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing the functionality of the loss functions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np


class TestStableSqrt(parameterized.TestCase):
  """Test class for the stable square root."""

  def test_stable_sqrt(self):
    """Tests calculation of the stable square root."""
    x = jnp.asarray([1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0.0])
    expected_y = jnp.sqrt(x)
    expected_dx = jnp.minimum(1 / (2 * expected_y), 1000.0)

    y, dx = jax.jvp(
        kfac_jax.utils.stable_sqrt,
        [x],
        [jnp.ones_like(x)],
    )

    np.testing.assert_allclose(y, expected_y)
    np.testing.assert_allclose(dx, expected_dx)

    y, vjp = jax.vjp(kfac_jax.utils.stable_sqrt, x)
    [dx] = vjp(jnp.ones_like(x))

    np.testing.assert_allclose(y, expected_y)
    np.testing.assert_allclose(dx, expected_dx)


if __name__ == "__main__":
  absltest.main()
