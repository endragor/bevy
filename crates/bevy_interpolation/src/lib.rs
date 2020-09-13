use bevy_math::{Mat4, Quat, Rect, Size, Vec2, Vec3, Vec4};
use bevy_transform::components::Transform;

/// Spherical linear interpolation
pub trait Slerp {
    fn interpolate(&self, other: &Self, t: f32) -> Self;
}

/// Linear interpolation
pub trait Lerp {
    fn interpolate(&self, other: &Self, t: f32) -> Self;
}

pub trait CustomInterpolation {
    fn interpolate(&self, other: &Self, t: f32) -> Self;
}

/// Rotation is interpolated spherically, while translation and scale are interpolated linearly.
impl CustomInterpolation for Transform {
    fn interpolate(&self, other: &Self, t: f32) -> Self {
        let (scale, rotation, translation) = self.value.to_scale_rotation_translation();
        let (other_scale, other_rotation, other_translation) =
            other.value.to_scale_rotation_translation();

        let new_scale = Lerp::interpolate(&scale, &other_scale, t);
        let new_rotation = Slerp::interpolate(&rotation, &other_rotation, t);
        let new_translation = Lerp::interpolate(&translation, &other_translation, t);

        Transform {
            value: Mat4::from_scale_rotation_translation(new_scale, new_rotation, new_translation),
            sync: self.sync,
        }
    }
}

pub trait Interpolatable {
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, scalar: f32) -> Self;
    fn dot(&self, other: &Self) -> f32;
}

impl Interpolatable for f32 {
    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn mul(&self, scalar: f32) -> Self {
        self * scalar
    }

    fn dot(&self, other: &Self) -> f32 {
        self * other
    }
}

impl Interpolatable for f64 {
    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn mul(&self, scalar: f32) -> Self {
        self * (scalar as f64)
    }

    fn dot(&self, other: &Self) -> f32 {
        (self * other) as f32
    }
}

impl Interpolatable for u32 {
    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn mul(&self, scalar: f32) -> Self {
        ((*self as f32) * scalar) as u32
    }

    fn dot(&self, other: &Self) -> f32 {
        (self * other) as f32
    }
}

impl Interpolatable for u64 {
    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn mul(&self, scalar: f32) -> Self {
        ((*self as f32) * scalar) as Self
    }

    fn dot(&self, other: &Self) -> f32 {
        (self * other) as f32
    }
}

impl Interpolatable for usize {
    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn mul(&self, scalar: f32) -> Self {
        ((*self as f32) * scalar) as Self
    }

    fn dot(&self, other: &Self) -> f32 {
        (self * other) as f32
    }
}

impl Interpolatable for i32 {
    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn mul(&self, scalar: f32) -> Self {
        ((*self as f32) * scalar) as Self
    }

    fn dot(&self, other: &Self) -> f32 {
        (self * other) as f32
    }
}

impl Interpolatable for i64 {
    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn mul(&self, scalar: f32) -> Self {
        ((*self as f32) * scalar) as Self
    }

    fn dot(&self, other: &Self) -> f32 {
        (self * other) as f32
    }
}

impl Interpolatable for isize {
    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn mul(&self, scalar: f32) -> Self {
        ((*self as f32) * scalar) as Self
    }

    fn dot(&self, other: &Self) -> f32 {
        (self * other) as f32
    }
}

impl Lerp for Quat {
    fn interpolate(&self, other: &Self, t: f32) -> Self {
        self.lerp(*other, t)
    }
}

impl Slerp for Quat {
    fn interpolate(&self, other: &Self, t: f32) -> Self {
        self.slerp(*other, t)
    }
}

impl Interpolatable for Size<f32> {
    fn add(&self, other: &Self) -> Self {
        Size::new(self.width + other.width, self.height + other.height)
    }

    fn mul(&self, scalar: f32) -> Self {
        Size::new(self.width * scalar, self.height * scalar)
    }

    fn dot(&self, other: &Self) -> f32 {
        self.width * other.width + self.height * other.height
    }
}

impl Interpolatable for Size<f64> {
    fn add(&self, other: &Self) -> Self {
        Size::new(self.width + other.width, self.height + other.height)
    }

    fn mul(&self, scalar: f32) -> Self {
        let scalar = scalar as f64;
        Size::new(self.width * scalar, self.height * scalar)
    }

    fn dot(&self, other: &Self) -> f32 {
        (self.width * other.width + self.height * other.height) as f32
    }
}

impl Interpolatable for Rect<f32> {
    fn add(&self, other: &Self) -> Self {
        Rect {
            left: self.left + other.left,
            right: self.right + other.right,
            top: self.top + other.top,
            bottom: self.bottom + other.bottom,
        }
    }

    fn mul(&self, scalar: f32) -> Self {
        Rect {
            left: self.left * scalar,
            right: self.right * scalar,
            top: self.top * scalar,
            bottom: self.bottom * scalar,
        }
    }

    fn dot(&self, other: &Self) -> f32 {
        self.left * other.left
            + self.right * other.right
            + self.top * other.top
            + self.bottom * other.bottom
    }
}

impl Interpolatable for Rect<f64> {
    fn add(&self, other: &Self) -> Self {
        Rect {
            left: self.left + other.left,
            right: self.right + other.right,
            top: self.top + other.top,
            bottom: self.bottom + other.bottom,
        }
    }

    fn mul(&self, scalar: f32) -> Self {
        let scalar = scalar as f64;
        Rect {
            left: self.left * scalar,
            right: self.right * scalar,
            top: self.top * scalar,
            bottom: self.bottom * scalar,
        }
    }

    fn dot(&self, other: &Self) -> f32 {
        (self.left * other.left
            + self.right * other.right
            + self.top * other.top
            + self.bottom * other.bottom) as f32
    }
}

impl Interpolatable for Vec2 {
    fn add(&self, other: &Self) -> Self {
        *self + *other
    }

    fn mul(&self, scalar: f32) -> Self {
        *self * scalar
    }

    fn dot(&self, other: &Self) -> f32 {
        Vec2::dot(*self, *other)
    }
}

impl Interpolatable for Vec3 {
    fn add(&self, other: &Self) -> Self {
        *self + *other
    }

    fn mul(&self, scalar: f32) -> Self {
        *self * scalar
    }

    fn dot(&self, other: &Self) -> f32 {
        Vec3::dot(*self, *other)
    }
}

impl Interpolatable for Vec4 {
    fn add(&self, other: &Self) -> Self {
        *self + *other
    }

    fn mul(&self, scalar: f32) -> Self {
        *self * scalar
    }

    fn dot(&self, other: &Self) -> f32 {
        Vec4::dot(*self, *other)
    }
}

impl<T: Interpolatable> Lerp for T {
    fn interpolate(&self, other: &Self, t: f32) -> Self {
        self.mul(1f32 - t).add(&other.mul(t))
    }
}

fn slerp_for_dot<T: Interpolatable>(left: &T, right: &T, t: f32, dot: f32) -> T {
    const DOT_THRESHOLD: f32 = 0.9995;

    if dot > DOT_THRESHOLD {
        // lerp
        left.mul(1f32 - t).add(&right.mul(t))
    } else {
        let dot_clamped = if dot < -1f32 {
            -1f32
        } else if dot > 1f32 {
            1f32
        } else {
            dot
        };
        let theta = dot_clamped.acos();
        let theta_sin = theta.sin();
        left.mul(((1f32 - t) * theta).sin() * theta_sin.recip())
            .add(&right.mul((t * theta).sin() * theta_sin.recip()))
    }
}

impl<T: Interpolatable> Slerp for T {
    fn interpolate(&self, other: &Self, t: f32) -> Self {
        let dot = self.dot(other);
        if dot < 0. {
            slerp_for_dot(self, &other.mul(-1f32), t, -dot)
        } else {
            slerp_for_dot(self, other, t, dot)
        }
    }
}
