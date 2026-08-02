use crate::models::{DruckerPrager, DruckerPragerPlasticState, ElasticCoefficients};
use bytemuck::{Pod, Zeroable};

/// Failure state of a [`RockModel`] particle.
#[derive(Copy, Clone, PartialEq, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct RockFailureState {
    /// Damage: `0.0` while the particle is intact (elastic), `1.0` once it failed
    /// (granular). Set by the GPU and never decreasing.
    pub damage: f32,
}

/// Elastic-brittle rock that behaves as a linear elastic solid until it fails,
/// after which it switches to Drucker-Prager plasticity.
///
/// Failure happens when stress reaches the failure envelope defined by the tensile cutoff
/// and Mohr-Coulomb criterion. Once the envelope is crossed, the particle is damaged
/// permanently and switches to the Drucker-Prager plasticity model.
///
/// The [`elastic`](Self::elastic) coefficients are used for the stress in both states.
/// The Lamé parameters stored in [`plastic`](Self::plastic) are only used by the plastic
/// projection itself.
#[derive(Copy, Clone, PartialEq, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct RockModel {
    /// Plastic state, only used after the particle has failed.
    pub plastic_state: DruckerPragerPlasticState,
    /// Whether this particle already failed.
    pub failure: RockFailureState,
    /// Maximum principal Kirchhoff stress the intact rock can carry, in Pa.
    ///
    /// Non-positive values disable the tensile cutoff.
    pub tensile_strength: f32,
    /// Uniaxial compressive strength of the intact rock, in Pa.
    ///
    /// Non-positive values disable the Mohr-Coulomb criterion.
    pub compressive_strength: f32,
    /// Drucker-Prager plasticity parameters, used after the particle has failed.
    pub plastic: DruckerPrager,
    /// Lamé parameters for the elastic coefficients, used in both intact and failed states.
    pub elastic: ElasticCoefficients,
}

impl RockModel {
    /// Creates a rock model.
    ///
    /// # Arguments
    ///
    /// * `young_modulus` - Elastic Young's modulus (Pa). Rocks are typically in the 1e9 - 5e10 range.
    /// * `poisson_ratio` - Elastic Poisson's ratio (0.0 - 0.5). Typically ~0.25 for rock.
    /// * `tensile_strength` - Maximum principal tensile stress before failure (Pa).
    /// * `compressive_strength` - Uniaxial compressive strength (Pa). Rock is much stronger in
    ///   compression than in tension, typically by 10-20x.
    pub fn new(
        young_modulus: f32,
        poisson_ratio: f32,
        tensile_strength: f32,
        compressive_strength: f32,
    ) -> Self {
        Self {
            plastic_state: DruckerPragerPlasticState::default(),
            failure: RockFailureState::default(),
            tensile_strength,
            compressive_strength,
            plastic: DruckerPrager::new(young_modulus, poisson_ratio),
            elastic: ElasticCoefficients::from_young_modulus(young_modulus, poisson_ratio),
        }
    }

    /// Returns `true` if this particle already failed (only meaningful for a model
    /// read back from the GPU).
    pub fn is_broken(&self) -> bool {
        self.failure.damage >= 0.5
    }

    /// Sets the friction angle (radians) of the broken material.
    ///
    /// This is also the angle used by the Mohr-Coulomb failure criterion of the intact rock.
    pub fn set_friction_angle(&mut self, angle: f32) {
        self.plastic.h0 = angle;
    }
}
