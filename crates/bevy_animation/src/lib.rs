use core::any::Any;
use std::{cmp::Ordering, collections::HashMap};

use bevy_app::{AppBuilder, Plugin};
use bevy_asset::{Assets, Handle, HandleId};
use bevy_core::Time;
use bevy_ecs::{Entity, IntoThreadLocalSystem, Resources, TypeInfo, World};
use bevy_interpolation::{CustomInterpolation, Lerp, Slerp};
use bevy_tasks::{TaskPool, TaskPoolBuilder};
use bevy_transform::components::Transform;

use dashmap::DashMap;

pub struct Keyframe<T> {
    pub time: f32,
    pub value: T,
}

/// A linearly interpolated track.
/// A valid track must have keyframes in strictly increasing order of `time`.
pub struct LerpTrack<T>
where
    T: Lerp + Clone,
{
    pub keyframes: Vec<Keyframe<T>>,
}

/// A spherically interpolated track.
/// A valid track must have keyframes in strictly increasing order of `time`.
pub struct SlerpTrack<T>
where
    T: Slerp + Clone,
{
    pub keyframes: Vec<Keyframe<T>>,
}

/// Uninterpolated track. The component is set to the last value.
/// A valid track must have keyframes in strictly increasing order of `time`.
pub struct StepTrack<T>
where
    T: Clone,
{
    pub keyframes: Vec<Keyframe<T>>,
}

/// Transform track.
/// Rotation is interpolated spherically, while translation and scale are interpolated linearly.
pub struct TransformTrack {
    pub keyframes: Vec<Keyframe<Transform>>,
}

pub struct CustomTrack<T>
where
    T: CustomInterpolation + Clone,
{
    pub keyframes: Vec<Keyframe<T>>,
}

pub struct CustomFnTrack<T, F>
where
    F: Fn(&T, &T, f32) -> T,
    T: Clone,
{
    pub keyframes: Vec<Keyframe<T>>,
    pub interpolation: F,
}

pub enum TrackState {
    Playing,
    Finished,
}

pub trait Track {
    fn update_component(&self, time: f32, component: &mut dyn Any) -> TrackState;
    fn type_info(&self) -> TypeInfo;
}

fn update_component<T, F>(
    keyframes: &[Keyframe<T>],
    time: f32,
    component: &mut T,
    interpolation: F,
) -> TrackState
where
    F: Fn(&T, &T, f32) -> T,
    T: Clone,
{
    let search_result = keyframes.binary_search_by(|x| {
        if x.time == time {
            Ordering::Equal
        } else if x.time < time {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });

    let start_idx = match search_result {
        Ok(idx) => idx,
        Err(0) => {
            return if keyframes.is_empty() {
                TrackState::Finished
            } else {
                TrackState::Playing
            }
        }
        Err(idx) => idx,
    };

    if start_idx >= keyframes.len() - 1 {
        *component = keyframes[start_idx].value.clone();
        TrackState::Finished
    } else {
        let key_start = &keyframes[start_idx];
        let key_end = &keyframes[start_idx + 1];
        let t = (time - key_start.time) / (key_end.time - key_start.time);
        *component = interpolation(&key_start.value, &key_end.value, t);
        TrackState::Playing
    }
}

fn step_interpolate<T>(start: &T, _end: &T, _t: f32) -> T
where
    T: Clone,
{
    (*start).clone()
}

impl<T> Track for LerpTrack<T>
where
    T: Lerp + Clone + 'static,
{
    fn update_component(&self, time: f32, component: &mut dyn Any) -> TrackState {
        let component = component.downcast_mut::<T>().unwrap();
        update_component(&self.keyframes, time, component, Lerp::interpolate)
    }

    fn type_info(&self) -> TypeInfo {
        TypeInfo::of::<T>()
    }
}

impl<T> Track for SlerpTrack<T>
where
    T: Slerp + Clone + 'static,
{
    fn update_component(&self, time: f32, component: &mut dyn Any) -> TrackState {
        let component = component.downcast_mut::<T>().unwrap();
        update_component(&self.keyframes, time, component, Slerp::interpolate)
    }

    fn type_info(&self) -> TypeInfo {
        TypeInfo::of::<T>()
    }
}

impl<T: Clone + 'static> Track for StepTrack<T> {
    fn update_component(&self, time: f32, component: &mut dyn Any) -> TrackState {
        let component = component.downcast_mut::<T>().unwrap();
        update_component(&self.keyframes, time, component, step_interpolate)
    }

    fn type_info(&self) -> TypeInfo {
        TypeInfo::of::<T>()
    }
}

impl<T> Track for CustomTrack<T>
where
    T: CustomInterpolation + Clone + 'static,
{
    fn update_component(&self, time: f32, component: &mut dyn Any) -> TrackState {
        let component = component.downcast_mut::<T>().unwrap();
        update_component(
            &self.keyframes,
            time,
            component,
            CustomInterpolation::interpolate,
        )
    }

    fn type_info(&self) -> TypeInfo {
        TypeInfo::of::<T>()
    }
}

impl<T, F> Track for CustomFnTrack<T, F>
where
    F: Fn(&T, &T, f32) -> T,
    T: Clone + 'static,
{
    fn update_component(&self, time: f32, component: &mut dyn Any) -> TrackState {
        let component = component.downcast_mut::<T>().unwrap();
        update_component(&self.keyframes, time, component, &self.interpolation)
    }

    fn type_info(&self) -> TypeInfo {
        TypeInfo::of::<T>()
    }
}

pub struct Animation {
    pub tracks: Vec<Box<dyn Track + Send + Sync>>,
    pub duration: f32,
}

pub struct AnimationManager {
    active_animations: DashMap<(HandleId, Entity), f32>,
    task_pool: TaskPool,
}

pub enum AnimationStatus {
    Playing(f32),
    NotPlaying,
}

impl AnimationManager {
    fn new() -> Self {
        Self {
            active_animations: DashMap::new(),
            task_pool: TaskPoolBuilder::new()
                .thread_name("Animation".to_owned())
                .build(),
        }
    }

    /// Requests to start playing the animation on the entity.
    /// The playing will not start until the animation is fully loaded. So if you
    /// need it to happen immediately, it's up to you to ensure the asset is already
    /// loaded.
    ///
    /// If the animation is already playing, it restarts.
    pub fn play(&self, animation: Handle<Animation>, entity: Entity) {
        *self
            .active_animations
            .entry((animation.id, entity))
            .or_insert(0f32) = 0f32;
    }

    /// Stops playing the animation on the entity.
    /// Does nothing if the animation is not playing.
    pub fn stop(&self, animation_id: HandleId, entity: Entity) {
        self.active_animations.remove(&(animation_id, entity));
    }

    pub fn advance_by(&self, animation_id: HandleId, entity: Entity, by: f32) {
        if let Some(mut time) = self.active_animations.get_mut(&(animation_id, entity)) {
            *time += by;
        }
    }

    pub fn advance_to(&self, animation_id: HandleId, entity: Entity, to: f32) {
        if let Some(mut time) = self.active_animations.get_mut(&(animation_id, entity)) {
            *time = to;
        }
    }

    /// Returns animation status for the given `animation` and `entity`.
    /// Due to the parallel nature of Bevy, this status may be outdated if other
    /// systems updated the entity at the same time.
    /// However, it is guaranteed to return most recent status for changes made
    /// within a single system.
    pub fn get_animation_status(&self, animation: HandleId, entity: Entity) -> AnimationStatus {
        if let Some(time) = self.active_animations.get(&(animation, entity)) {
            AnimationStatus::Playing(*time)
        } else {
            AnimationStatus::NotPlaying
        }
    }
}

/// Returns `true` if the animation should continue.
fn step_animation(
    world: &World,
    anim_handle: HandleId,
    entity: Entity,
    assets: &Assets<Animation>,
    last_time: f32,
    delta: f32,
) -> bool {
    if let Ok(entity) = world.entity(entity) {
        if let Some(animation) = assets.get_with_id(anim_handle) {
            let time = last_time + delta;
            let mut all_tracks_finished = true;
            for track in &animation.tracks {
                // Safe because this runs from a thread-local system that groups animations
                // by entities into one task. So no overlaps may occur.
                let component = unsafe { entity.get_unchecked_mut_any(&track.type_info()) };
                if let Some(component) = component {
                    match track.update_component(time, component) {
                        TrackState::Playing => all_tracks_finished = false,
                        TrackState::Finished => {}
                    }
                }
            }
            time < animation.duration && !all_tracks_finished
        } else {
            // If the animation never started playing, wait for the asset.
            // Otherwise it seems the asset was removed, so stop playing.
            //
            // TODO: not sure if this is the correct way to handle the unreliable
            // nature of assets. It'd be better if they were reference-counted.
            last_time == 0f32
        }
    } else {
        // The entity does not exist. Stop playing.
        false
    }
}

fn animation_system(world: &mut World, resources: &mut Resources) {
    let manager = resources.get_mut::<AnimationManager>().unwrap();
    let delta = resources.get::<Time>().unwrap().delta.as_secs_f32();
    let assets = resources.get::<Assets<Animation>>().unwrap();
    // TODO: reduce allocations
    let mut grouped_by_entity = HashMap::with_capacity(manager.active_animations.len());
    for entry in manager.active_animations.iter() {
        let (anim_handle, entity) = entry.key();
        let time = entry.value();
        grouped_by_entity
            .entry(*entity)
            .or_insert_with(Vec::new)
            .push((*anim_handle, *time));
    }
    let active_animations = &manager.active_animations;
    manager.task_pool.scope(|s| {
        for (entity, animations) in grouped_by_entity.into_iter() {
            let world: &World = world;
            let assets: &Assets<Animation> = &*assets;
            s.spawn(async move {
                for (anim_handle, cur_time) in animations.into_iter() {
                    if !step_animation(world, anim_handle, entity, assets, cur_time, delta) {
                        active_animations.remove(&(anim_handle, entity));
                    } else {
                        if let Some(mut time) = active_animations.get_mut(&(anim_handle, entity)) {
                            *time = cur_time + delta;
                        }
                    }
                }
            });
        }
    });
}

pub struct AnimationPlugin;

impl Default for AnimationPlugin {
    fn default() -> Self {
        AnimationPlugin
    }
}

pub mod stage {
    pub const ANIMATION: &str = "animation";
}

impl Plugin for AnimationPlugin {
    fn build(&self, app: &mut AppBuilder) {
        // TODO: not sure if this is the right place for animation stage
        app.add_stage_before(bevy_app::stage::POST_UPDATE, stage::ANIMATION)
            .add_resource(AnimationManager::new());

        app.add_system_to_stage(stage::ANIMATION, animation_system.thread_local_system());
    }
}

#[cfg(test)]
mod test {}
