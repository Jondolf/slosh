use slosh_testbed3d::{AppState, PhysicsContext, PhysicsState, RapierData, slosh};

use glam::{Vec3, Vec4, vec3};
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder};
use slang_hal::backend::{Backend, WebGpu};
use slosh::{
    pipeline::MpmData,
    solver::{GpuBoundaryCondition, Particle, ParticleModel, SimulationParams},
};

#[expect(dead_code)]
fn main() {
    panic!("Run the `testbed3` example instead.");
}

const DENSITY: f32 = 2700.0;
const YOUNG_MODULUS: f32 = 5.0e10;
const POISSON_RATIO: f32 = 0.25;

const ROCK_TENSILE_STRENGTH: f32 = 5.0e9;
const ROCK_COMPRESSIVE_STRENGTH: f32 = 5.0e10;
const JOINT_TENSILE_STRENGTH: f32 = 1.0e7;
const JOINT_COMPRESSIVE_STRENGTH: f32 = 1.0e8;

const GRAVITY_SCALE: f32 = 4.0;
const DAMPING: f32 = 0.0;

const BENCH_HALF_LENGTH: f32 = 23.0;
const BENCH_HALF_DEPTH: f32 = 10.0;
const BENCH_HEIGHT: f32 = 25.0;
const UNDERCUT_HEIGHT: f32 = 22.0;
const UNDERCUT_START: f32 = 20.0;
const ABUTMENT: f32 = 0.0;

const JOINT_PERSISTENCE: f32 = 0.7;
const JOINT_PATCH: f32 = 0.7;
const JOINT_JITTER: f32 = 0.2;
const JOINT_HALF_WIDTH: f32 = 0.2;
const JOINT_WIDTH_NOISE: f32 = 0.2;
const JOINT_NOISE_SCALE: f32 = 1.8;

const CELL_WIDTH: f32 = 0.6;
const PARTICLES_PER_CELL_DIM: usize = 2;
const NUM_SUBSTEPS: u32 = 150;
const GRAVITY_RAMP_FRAMES: f32 = 90.0;

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn hash01(a: i32, b: i32, c: i32) -> f32 {
    let h = hash_u32(a as u32 ^ hash_u32((b as u32).wrapping_add(hash_u32(c as u32))));
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Trilinearly interpolated value noise in `[0, 1]`.
fn value_noise(p: Vec3, salt: i32) -> f32 {
    let cell = p.floor();
    let f = p - cell;
    let w = f * f * (3.0 - 2.0 * f);
    let (cx, cy, cz) = (cell.x as i32, cell.y as i32, cell.z as i32);

    let mut acc = 0.0;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let corner = hash01(
                    cx + dx,
                    cy + dy,
                    (cz + dz).wrapping_mul(31).wrapping_add(salt),
                );
                let wx = if dx == 0 { 1.0 - w.x } else { w.x };
                let wy = if dy == 0 { 1.0 - w.y } else { w.y };
                let wz = if dz == 0 { 1.0 - w.z } else { w.z };
                acc += corner * wx * wy * wz;
            }
        }
    }
    acc
}

struct JointSet {
    normal: Vec3,
    spacing: f32,
    offset: f32,
    salt: i32,
}

impl JointSet {
    fn plane_pos(&self, p: i32) -> f32 {
        let jitter = (hash01(p, self.salt, 0) - 0.5) * 2.0 * JOINT_JITTER;
        (p as f32 + jitter) * self.spacing + self.offset
    }

    /// Index of the slab of rock containing `u`, the coordinate along the normal.
    fn slab(&self, u: f32) -> i32 {
        let guess = ((u - self.offset) / self.spacing).floor() as i32;
        for p in [guess + 1, guess] {
            if self.plane_pos(p) <= u {
                return p;
            }
        }
        guess - 1
    }

    fn distance(&self, u: f32) -> f32 {
        let p = self.slab(u);
        (u - self.plane_pos(p)).abs().min(self.plane_pos(p + 1) - u)
    }
}

pub fn rock_break_demo(backend: &WebGpu, app_state: &mut AppState) -> PhysicsContext {
    let mut rapier_data = RapierData::default();

    let spacing = CELL_WIDTH / PARTICLES_PER_CELL_DIM as f32;
    let radius = spacing / 2.0;
    // Never let the noise thin a seam below a single particle layer, or it stops separating.
    let min_half_width = spacing * 0.6;

    let nx = (2.0 * BENCH_HALF_LENGTH / spacing) as usize;
    let ny = (BENCH_HEIGHT / spacing) as usize;
    let nz = (2.0 * BENCH_HALF_DEPTH / spacing) as usize;

    // Two steep sets and a shallow bedding one.
    let joint_sets = [
        JointSet {
            normal: vec3(1.0, 0.55, 0.12).normalize(),
            spacing: 5.0,
            offset: 0.0,
            salt: 1,
        },
        JointSet {
            normal: vec3(0.17, 0.09, 1.0).normalize(),
            spacing: 5.5,
            offset: 2.0,
            salt: 2,
        },
        JointSet {
            normal: vec3(-0.23, 1.0, 0.14).normalize(),
            spacing: 4.5,
            offset: 0.7,
            salt: 3,
        },
    ];

    let mut particles = vec![];
    let mut colors = vec![];

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let position = vec3(
                    (i as f32 + 0.5) * spacing - BENCH_HALF_LENGTH,
                    (j as f32 + 0.5) * spacing,
                    (k as f32 + 0.5) * spacing - BENCH_HALF_DEPTH,
                );

                // Mine out the undercut, leaving the rock above it overhanging.
                if position.y < UNDERCUT_HEIGHT && position.x > -UNDERCUT_START {
                    continue;
                }

                // Which block this belongs to, and whether it sits on an open stretch of a
                // joint plane rather than on a rock bridge.
                let patch = (position / JOINT_PATCH).floor();
                let mut block_id = 0x9e37_79b9u32;
                let mut jointed = false;

                for set in &joint_sets {
                    let u = position.dot(set.normal);
                    block_id ^= hash_u32(
                        (set.slab(u) as u32)
                            .wrapping_add((set.salt as u32).wrapping_mul(0x85eb_ca6b)),
                    );

                    let wobble = (value_noise(position / JOINT_NOISE_SCALE, set.salt) - 0.5) * 2.0;
                    let half_width =
                        (JOINT_HALF_WIDTH * (1.0 + JOINT_WIDTH_NOISE * wobble)).max(min_half_width);

                    if set.distance(u) < half_width {
                        jointed |=
                            hash01(patch.x as i32, patch.y as i32, patch.z as i32 + set.salt)
                                < JOINT_PERSISTENCE;
                    }
                }

                let model = if jointed {
                    ParticleModel::rock(
                        YOUNG_MODULUS,
                        POISSON_RATIO,
                        JOINT_TENSILE_STRENGTH,
                        JOINT_COMPRESSIVE_STRENGTH,
                    )
                } else {
                    ParticleModel::rock(
                        YOUNG_MODULUS,
                        POISSON_RATIO,
                        ROCK_TENSILE_STRENGTH,
                        ROCK_COMPRESSIVE_STRENGTH,
                    )
                };

                let mut particle = Particle::new(position, radius, DENSITY, model);
                particle.dynamics.set_damping(DAMPING);

                // The abutment stands for the rest of the rock mass, and the bottom layer anchors
                // the remaining leg of rock to the floor.
                let fixed = position.x < -BENCH_HALF_LENGTH + ABUTMENT || position.y < spacing;
                particle.dynamics.set_fixed(fixed);

                // One shade per block, so a chunk coming away in one piece reads as a single
                // object. Open joints are darkened into seams.
                colors.push(if fixed {
                    Vec4::new(0.22, 0.22, 0.26, 1.0)
                } else {
                    let id = block_id as i32;
                    let brightness = 0.55 + 0.75 * hash01(id, 11, 0);
                    let warmth = 0.9 + 0.25 * hash01(id, 22, 0);
                    let shade = if jointed { 0.55 } else { 1.0 };
                    Vec4::new(
                        0.55 * brightness * warmth * shade,
                        0.51 * brightness * shade,
                        0.47 * brightness / warmth * shade,
                        1.0,
                    )
                });
                particles.push(particle);
            }
        }
    }

    println!(
        "Rock break: {} particles of {:.2} m at {:.1} g.",
        particles.len(),
        spacing,
        GRAVITY_SCALE
    );

    if !app_state.restarting {
        app_state.min_num_substeps = NUM_SUBSTEPS;
        app_state.max_num_substeps = NUM_SUBSTEPS;
        app_state.gravity_factor = 1.0;
    }

    app_state.particle_colors = Some(colors);
    app_state.initial_camera_eye = Some([38.0, 24.0, 38.0]);
    app_state.initial_camera_target = Some([0.0, 5.0, 0.0]);

    let gravity = vec3(0.0, -9.81 * GRAVITY_SCALE, 0.0) * app_state.gravity_factor;
    let params = SimulationParams {
        gravity,
        dt: 1.0 / 60.0,
    };

    // Floor of the undercut, catching the caved rock.
    let rb = RigidBodyBuilder::fixed().translation(vec3(0.0, -2.0, 0.0));
    let rb_handle = rapier_data.bodies.insert(rb);
    let co = ColliderBuilder::cuboid(80.0, 2.0, 80.0);
    let co_handle =
        rapier_data
            .colliders
            .insert_with_parent(co, rb_handle, &mut rapier_data.bodies);
    let boundary_conditions = [(co_handle, GpuBoundaryCondition::stick())];

    let data = MpmData::new(
        backend,
        params,
        &particles,
        &rapier_data.bodies,
        &rapier_data.colliders,
        &boundary_conditions,
        CELL_WIDTH,
        60_000,
    )
    .unwrap();

    // Ramp gravity in rather than switching it on at full strength.
    let mut frame = 0u32;
    let ramp = move |state: &mut PhysicsState| {
        if frame as f32 >= GRAVITY_RAMP_FRAMES {
            return;
        }

        frame += 1;
        state.data.gravity = gravity * (frame as f32 / GRAVITY_RAMP_FRAMES);
        let params = SimulationParams {
            gravity: state.data.gravity,
            dt: state.data.base_dt / NUM_SUBSTEPS as f32,
        };
        state
            .backend
            .write_buffer(state.data.sim_params.params.buffer_mut(), 0, &[params])
            .unwrap();
    };

    PhysicsContext {
        data,
        rapier_data,
        callbacks: vec![Box::new(ramp)],
        hooks_state: None,
    }
}
