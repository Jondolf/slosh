//! Brittle rock model that is elastic while intact, or granular (Drucker-Prager) once it fails.

module rock;

import slosh.aliases;
import slosh.models.utils;
import slosh.models.interfaces;
import slosh.models.linear_elasticity;
import slosh.models.drucker_prager;
import stensor.geometry.svd2;
import stensor.geometry.svd_glam.svd3;

/// Per-particle failure state of the rock model.
public struct RockFailureState {
    /// Damage state of the particle.
    ///
    /// 0 = intact (elastic), 1 = failed (granular).
    public float damage;
}

/// Result of a rock model update.
public struct RockUpdateResult {
    public Mat<float> kirchoff_stress;
    public Mat<float> deformation_gradient;
    public DruckerPragerPlasticState plastic_state;
    public float damage;
}

/// Elastic-brittle rock that is elastic while intact, or granular (Drucker-Prager) once it fails.
public struct RockModel {
    DruckerPragerPlasticState plastic_state;
    RockFailureState failure;
    /// Maximum principal Kirchhoff stress the intact rock can carry, in Pa.
    ///
    /// Non-positive values disable the tensile cutoff.
    float tensile_strength;
    /// Uniaxial compressive strength of the intact rock, in Pa.
    ///
    /// Non-positive values disable the Mohr-Coulomb criterion.
    float compressive_strength;
    DruckerPragerPlasticity plastic;
    LinearElasticModel elastic;

    /// Byte offset of `plastic_state` within this struct.
    public func state_offset() -> uint {
        return 0;
    }

    /// Byte offset of `failure` within this struct.
    public func failure_offset() -> uint {
        return 12;
    }

    /// Principal Kirchhoff stresses of the intact (linear corotated elastic) response,
    /// computed from the singular values of the deformation gradient.
    ///
    /// In the principal frame of `F = U S V^T`, the corotated model gives
    /// `tau_i = 2 mu (s_i - 1) s_i + lambda (J - 1) J`. Positive values are tensile.
    public func principal_kirchoff_stress(singular_values: Vect<float>) -> Vect<float> {
        #if DIM == 2
        let j = singular_values.x * singular_values.y;
        #else
        let j = singular_values.x * singular_values.y * singular_values.z;
        #endif
        // The linear corotated model gives the Kirchhoff stress
        //
        // tau = 2 mu (F - R) F^T + lambda (J - 1) J I
        //
        // where F is the deformation gradient
        //       R is the rotation from the polar decomposition of F
        //       J is the determinant of F
        //
        // In the principal frame, tau is diagonal, giving the principal stresses
        // tau_i = 2 mu (s_i - 1) s_i + lambda (j - 1) j
        let deviatoric = 2.0 * this.elastic.mu * (singular_values - Vect<float>(1.0)) * singular_values;
        return deviatoric + Vect<float>(this.elastic.lambda * (j - 1.0) * j);
    }

    /// Tests whether the given principal stresses exceed the failure envelope
    /// defined by the tensile cutoff and Mohr-Coulomb criterion.
    ///
    /// Positive stress values are tensile, negative values are compressive.
    public func exceeds_failure_envelope(principal_stress: Vect<float>) -> bool {
        var max_stress = principal_stress[0];
        var min_stress = principal_stress[0];

        [[ForceUnroll]]
        for (var i = 1; i < DIM; i++) {
            max_stress = max(max_stress, principal_stress[i]);
            min_stress = min(min_stress, principal_stress[i]);
        }

        // Tensile failure
        if (this.tensile_strength > 0.0 && max_stress > this.tensile_strength) {
            return true;
        }

        // Shear failure (Mohr-Coulomb)
        //
        // Failure is sliding on an internal plane, resisted by friction.
        // On a plane with normal stress sigma_n (compression-positive), it occurs when
        //
        // |tau_shear| = c + sigma_n tan(phi)
        //
        // where c is the cohesion and phi is the internal friction angle.
        // (See https://en.wikipedia.org/wiki/Mohr-Coulomb_theory)
        //
        // Over all plane orientations, (sigma_n, tau) traces Mohr's circle,
        // centered at the mean stress s = (s1 + s3) / 2 with radius r = (s1 - s3) / 2.
        // Failure happens when the circle touches the line tau = c + sigma_n tan(phi),
        // which is given by the distance from the center to the line
        //
        // (s tan(phi) + c) / sqrt(tan(phi)^2 + 1) = r
        // <=> (c + s tan(phi)) cos(phi) = r
        // <=> c cos(phi) + (s1 + s3) / 2 * sin(phi) = (s1 - s3) / 2
        //
        // (See https://en.wikipedia.org/wiki/Mohr's_circle)
        //
        // Solving for s1 gives:
        //
        // s1 = N_phi * s3 + UCS
        //
        // where N_phi = (1 + sin(phi)) / (1 - sin(phi)) is how much a unit of confining
        // pressure raises the strength, and UCS = 2 c cos(phi) / (1 - sin(phi)) is the
        // uniaxial compressive strength.
        //
        // Our principal stresses are tension-positive, so s1 = -min_stress and
        // s3 = -max_stress, giving the final condition
        //
        // -min_stress > N_phi * (-max_stress) + UCS
        // <=> N_phi * max_stress - min_stress > UCS
        if (this.compressive_strength > 0.0) {
            let sin_phi = clamp(sin(this.plastic.ha), 0.0, 0.95);
            let n_phi = (1.0 + sin_phi) / (1.0 - sin_phi);
            if (n_phi * max_stress - min_stress > this.compressive_strength) {
                return true;
            }
        }

        return false;
    }

    /// Runs one constitutive update, using elastic response while intact,
    // or Drucker-Prager response once the failure envelope has been crossed.
    public func update_rock(deformation_gradient: Mat<float>) -> RockUpdateResult {
        var damage = this.failure.damage;

        if (damage < 1.0) {
            #if DIM == 2
            let svd = Svd2(deformation_gradient);
            #else
            let svd = Svd3(deformation_gradient);
            #endif

            let principal_stress = this.principal_kirchoff_stress(svd.S);

            if (!this.exceeds_failure_envelope(principal_stress)) {
                // Still intact, use the linear elastic response.
                let stress = this.elastic.kirchoff_stress_from_svd(deformation_gradient, svd.U, svd.S, svd.Vt);
                return RockUpdateResult(stress, deformation_gradient, this.plastic_state, damage);
            }

            // The particle broke.
            damage = 1.0;
        }

        // Failed particle, use Drucker-Prager plasticity.
        let projection = this.plastic.project(this.plastic_state, deformation_gradient);
        let stress = this.elastic.kirchoff_stress(projection.deformation_gradient);
        return RockUpdateResult(stress, projection.deformation_gradient, projection.state, damage);
    }

    public func timestep_bound(
        particle_density0: float,
        particle_velocity: Vect<float>,
        particle_def_grad_det: float,
        cell_width: float,
    ) -> float {
        return this.elastic.timestep_bound(particle_density0, particle_velocity, particle_def_grad_det, 1.0, cell_width);
    }
}
