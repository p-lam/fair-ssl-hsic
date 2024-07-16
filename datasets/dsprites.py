"""Code adapted from https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/data/ground_truth/dsprites.py"""
import numpy as np
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
DSprites-Unfair Helpers
"""
class GroundTruthData(object):
    """Abstract class for data sets that are two-step generative models."""
    @property
    def num_factors(self):
        raise NotImplementedError()

    @property
    def factors_num_values(self):
        raise NotImplementedError()

    @property
    def observation_shape(self):
        raise NotImplementedError()

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        raise NotImplementedError()

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def sample_observations(self, num, random_state):
        """Sample a batch of observations X."""
        return self.sample(num, random_state)[1]
    
class DSprites(GroundTruthData):
    """
    DSprites dataset.

    The data set was originally introduced in "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework" and can be downloaded from
    https://github.com/deepmind/dsprites-dataset.

    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """
    def __init__(self, root_dir, latent_factor_indices=None):
        # by default, all factors (including shape) are ground truth factors
        if latent_factor_indices is None:
            latent_factor_indices = list(range(6))
        self.latent_factor_indices = latent_factor_indices
        self.data_shape = [64, 64, 1]
        # load the data so we can sample from it 
        dsprites_path = root_dir + '/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        data = np.load(dsprites_path, encoding="latin1", allow_pickle=True)
        self.images = np.array(data["imgs"])
        self.factor_sizes = np.array(data["metadata"][()]["latents_sizes"], dtype=np.int64)
        self.full_factor_sizes = [1, 3, 6, 40, 32, 32]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latent_factor_indices)
    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        return self.sample_observations_from_factors_no_color(factors, random_state)

    def sample_observations_from_factors_no_color(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return np.expand_dims(self.images[indices].astype(np.float32), axis=3)

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)

class ColorDSprites(DSprites):
    """Color DSprites.

    This data set is the same as the original DSprites data set except that when
    sampling the observations X, the sprite is colored in a randomly sampled
    color.

    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """

    def __init__(self, latent_factor_indices=None):
        DSprites.__init__(self, latent_factor_indices)
        self.data_shape = [64, 64, 3]

    def sample_observations_from_factors(self, factors, random_state):
        no_color_observations = self.sample_observations_from_factors_no_color(
            factors, random_state)
        observations = np.repeat(no_color_observations, 3, axis=3)
        color = np.repeat(
                np.repeat(
                    random_state.uniform(0.5, 1, [observations.shape[0], 1, 1, 3]),
                    observations.shape[1],
                    axis=1),
                observations.shape[2],
                axis=2)
        return observations * color
  
class SplitDiscreteStateSpace(object):
    """State space with factors split between latent variable and observations."""

    def __init__(self, factor_sizes, latent_factor_indices):
        self.factor_sizes = factor_sizes
        self.num_factors = len(self.factor_sizes)
        self.latent_factor_indices = latent_factor_indices
        self.observation_factor_indices = [
            i for i in range(self.num_factors)
            if i not in self.latent_factor_indices
        ]

    @property
    def num_latent_factors(self):
        return len(self.latent_factor_indices)

    def sample_latent_factors(self, num, random_state):
        """Sample a batch of the latent factors."""
        factors = np.zeros(
            shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
        for pos, i in enumerate(self.latent_factor_indices):
            factors[:, pos] = self._sample_factor(i, num, random_state)
        return factors

    def sample_all_factors(self, latent_factors, random_state):
        """Samples the remaining factors based on the latent factors."""
        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(
            shape=(num_samples, self.num_factors), dtype=np.int64)
        all_factors[:, self.latent_factor_indices] = latent_factors
        # Complete all the other factors
        for i in self.observation_factor_indices:
            all_factors[:, i] = self._sample_factor(i, num_samples, random_state)
        return all_factors

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)

# Object colors generated using
# >> seaborn.husl_palette(n_colors=6, h=0.1, s=0.7, l=0.7)
OBJECT_COLORS = np.array(
    [[0.9096231780824386, 0.5883403686424795, 0.3657680693481871],
     [0.6350181801577739, 0.6927729880940552, 0.3626904230371999],
     [0.3764832455369271, 0.7283900430001952, 0.5963114605342514],
     [0.39548987063404156, 0.7073922557810771, 0.7874577552076919],
     [0.6963644829189117, 0.6220697032672371, 0.899716387820763],
     [0.90815966835861, 0.5511103319168646, 0.7494337214212151]])

BACKGROUND_COLORS = np.array([
    (0., 0., 0.),
    (.25, .25, .25),
    (.5, .5, .5),
    (.75, .75, .75),
    (1., 1., 1.),
])

class AbstractDSprites(DSprites):
    """DSprites variation for abstract reasoning task.

    Rotation is not considered a ground-truth factor and we sample random colors
    both for the object and the background.

    The ground-truth factors of variation are (in the default setting):
    0 - background color (5 different values)
    1 - object color (6 different values)
    2 - shape (3 different values)
    3 - scale (6 different values)
    4 - position x (32 different values)
    5 - position y (32 different values)
    """

    def __init__(self):
        # We retain all original factors except shape.
        DSprites.__init__(self, [1, 2, 4, 5])
        self.data_shape = [64, 64, 3]

    @property
    def num_factors(self):
        return 2 + self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return ([BACKGROUND_COLORS.shape[0], OBJECT_COLORS.shape[0]] +
                [self.full_factor_sizes[i] for i in self.latent_factor_indices])

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        colors = np.zeros((num, 2), dtype=np.int64)
        colors[:, 0] = random_state.randint(BACKGROUND_COLORS.shape[0], size=num)
        colors[:, 1] = random_state.randint(OBJECT_COLORS.shape[0], size=num)
        other_factors = self.state_space.sample_latent_factors(num, random_state)
        return np.concatenate([colors, other_factors], axis=-1)

    def sample_observations_from_factors(self, factors, random_state):
        mask = self.sample_observations_from_factors_no_color(
            factors[:, 2:], random_state)

        background_color = BACKGROUND_COLORS[factors[:, 0]]
        object_color = OBJECT_COLORS[factors[:, 1]]

        # Add axis for height and width.
        background_color = np.expand_dims(np.expand_dims(background_color, 1), 1)
        object_color = np.expand_dims(np.expand_dims(object_color, 1), 1)

        return mask * object_color + (1. - mask) * background_color