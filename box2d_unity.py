#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Box2D-Packed contributors
# SPDX-License-Identifier: MIT
"""
box2d_unity: Box2D-Packed as the 2D physics backend for crust's tools/unity_pack.py.

unity_pack packs a Unity-shaped project into engine.c / data.c. Its data tables describe every
Rigidbody2D (_Rigidbody2D_*) and Collider2D (_Collider2D_*), and its engine sends
OnCollisionEnter2D / Stay2D / Exit2D by comparing each fixed step's touching pairs with the
previous step's. This module is unity_pack's 2D physics: it generates a Box2D world for those
tables and keeps everything else:

    engine_physics_fixed()                 (engine.c, unity_pack)
        engine_box2d_step()                (physics_box2d.c, generated here)
            push script changes: velocity, teleports, gravity scale, damping, world gravity
            b2World_Step
            pull positions and velocities into the packed tables
            report touching pairs with engine_col2d_contact()
        engine_physics_collide2d_messages() (unity_pack: Enter / Stay / Exit, after the step)

Touching pairs come from Box2D contact begin and end events. With `--physics-inject`, they come
from box2d_pack injection markers instead: the injected code only records the pair, so scripts
still run after the step, as in Unity.

engine.c exports what the glue needs, in unity_pack's C subset:

    void engine_rb2d_get_pos( int rb, float* x, float* y );   Rigidbody2D owner position
    void engine_rb2d_set_pos( int rb, float x, float y );
    void engine_col2d_center( int ci, float* x, float* y );  collider world center
    void engine_col2d_contact( int a, int b );               report a touching pair

With plan["physics2d_live"] (unity_pack sets it when some GameObjects can leave the simulation,
e.g. scenes that are not loaded, or objects destroyed -- Destroy, Godot's QueueFree) it also
exports

    int engine_rb2d_live( int rb );                          owner in the simulation
    int engine_col2d_live( int ci );                         static collider in the simulation

and bodies whose owner is not live are disabled (b2Body_Disable) until it is again.

It always exports Physics2D.Raycast / OverlapCircle / OverlapPoint for the engine (a collider
index, -1 for none; see QUERY_FUNCTIONS); with plan["physics2d_rotation"] bodies turn (see
_with_rotation) instead of having their rotation locked.

With plan["physics2d_joints"] it builds unity_pack's 2D joints (HingeJoint2D, DistanceJoint2D,
SpringJoint2D, FixedJoint2D, SliderJoint2D, WheelJoint2D; see _with_joints) and reports a broken
one with

    void engine_joint2d_broken( int j );

With plan["physics2d_contacts"] (scripts read Collision2D contacts) each touching pair's manifold
is reported just before the pair, from b2Shape_GetContactData:

    void engine_col2d_manifold( int a, int b, float nx, float ny, int n,
                                float p0x, float p0y, float p1x, float p1y );
                                normal from a to b, n (0-2) world contact points

Mapping:
    Rigidbody2D Dynamic / Kinematic / Static  ->  b2_dynamicBody / kinematic / static; rotation
                                                  locked unless plan["physics2d_rotation"],
                                                  and then locked by FreezeRotation only
    BoxCollider2D / CircleCollider2D          ->  offset box / circle, rotated by the collider
    CapsuleCollider2D                         ->  capsule along m_Direction (circle when short)
    PolygonCollider2D                         ->  a triangle polygon per triangle of its paths
                                                  (plan["physics2d_polygons"], see _with_polygons)
    Collider2D without a Rigidbody2D          ->  static body at the collider center
    m_IsTrigger                               ->  sensor, no collision messages; with
                                                  plan["physics2d_triggers"] its overlaps are
                                                  reported with engine_col2d_trigger( a, b )
                                                  for OnTrigger*2D (injected too with
                                                  --physics-inject)
    Rigidbody2D mass                          ->  body mass (and again when a script changes it)
    Rigidbody2D bodyType / isKinematic        ->  body type, and b2Body_SetType when a script
                                                  changes it
    friction / bounciness + combine modes     ->  world friction / restitution callbacks that
                                                  apply Unity's PhysicsMaterialCombine rules

Godot mode (`emit_glue( ..., mode="godot" )`, for crust's tools/godot_pack.py, which packs a
Godot 4 project through the same tables). The tables and the step are the same; what differs is
what Godot means by them:

    units                     ->  Godot's pixels, y down, as the scene and scripts have them: the
                                  world runs in them, with b2SetLengthUnitsPerMeter scaling Box2D's
                                  tolerances (length_units_per_meter, default 64)
    PhysicsMaterial           ->  Godot's rules: friction |min(a, b)|, bounce clamp(a + b, 0, 1),
                                  a rough material's friction and an absorbent material's bounce
                                  counted negative (rough / absorbent in the combine columns)
    linear_damp               ->  Godot damps once a step, v *= max(0, 1 - dt * d); the table holds
                                  that d, and the glue sets the Box2D damping whose substeps
                                  compound to the same factor
    fixed step                ->  1/60 s when Time_fixedDeltaTime is unset (physics ticks 60)
    collision layers          ->  with plan["physics2d_layers"], Godot's collision_layer /
                                  collision_mask (_Collider2D_layer_bits / _mask_bits): a pair
                                  collides when a dynamic side's mask has the other's layer
                                  (the side Godot pushes); areas' overlaps are godot_pack's to
                                  filter (_with_godot_layers)
    Area2D (a sensor)         ->  overlaps: every shape takes sensor events, and a sensor's begin
                                  and end touch report the pair with engine_col2d_contact, as a
                                  contact does -- godot_pack sends body_entered / area_entered from
                                  them. Unity mode's triggers stay silent, as before
"""

import json
import os
import sys

__all__ = ["emit_glue", "build_library", "GLUE_FILE", "INJECT_FILE", "MODES"]

GLUE_FILE = "physics_box2d.c"
INJECT_FILE = "box2d_inject.json"

_HERE = os.path.dirname( os.path.abspath( __file__ ) )

#: The engines whose tables the glue can take. unity is the default.
MODES = ( "unity", "godot" )


def _counts( plan ):
    """Rigidbody2D capacity (authored + AddComponent budget) and collider count."""
    budget = plan.get( "addcomponent_budget" ) or {}
    rb = len( plan.get( "rigidbody2d" ) or [] ) + int( budget.get( "Rigidbody2D" ) or 0 )
    col = len( plan.get( "collider2d" ) or [] )
    return rb, col


def emit_glue( outdir, plan, inject=False, sub_steps=4, mode="unity", length_units_per_meter=64.0 ):
    """
    Write physics_box2d.c (and box2d_inject.json when inject) into outdir. Returns the paths.
    mode is "unity" (unity_pack) or "godot" (godot_pack); length_units_per_meter is Godot's
    pixels per Box2D metre, and is ignored in unity mode, whose units are metres.
    """
    if mode not in MODES:
        raise ValueError( "box2d_unity: unknown mode %r (one of %s)" % ( mode, ", ".join( MODES ) ) )
    if mode == "godot" and not float( length_units_per_meter ) > 0.0:
        raise ValueError( "box2d_unity: length_units_per_meter must be positive" )
    rb_count, col_count = _counts( plan )
    n_rb = max( 1, rb_count )
    n_col = max( 1, col_count )
    max_pairs = max( 1, col_count * ( col_count - 1 ) // 2 )

    if col_count > 0:
        collider_decls = COLLIDER_EXTERNS
    else:
        # unity_pack emits no collider tables for a scene without colliders
        collider_decls = COLLIDER_EXTERNS.replace( "extern const", "static const" ).replace(
            "_Collider2D_count;", "_Collider2D_count = 0;" ).replace( "[];", "[1];" )
    glue = GLUE_TEMPLATE.format(
        COLLIDER_DECLS=collider_decls,
        N_RB=n_rb,
        N_COL=n_col,
        MAX_PAIRS=max_pairs,
        SUB_STEPS=int( sub_steps ),
        **_mode_parts( mode, length_units_per_meter ),
    )
    if plan.get( "physics2d_live" ):
        glue = _with_live_gate( glue )
    if plan.get( "physics2d_contacts" ):
        glue = _with_contact_manifolds( glue )
    if plan.get( "physics2d_triggers" ) and mode == "unity":
        glue = _with_unity_triggers( glue )
    glue += QUERY_FUNCTIONS
    if plan.get( "physics2d_layers" ) and mode == "godot":
        glue = _with_godot_layers( glue )
    if plan.get( "physics2d_rotation" ) and mode == "unity":
        glue = _with_rotation( glue )
    if plan.get( "physics2d_joints" ) and mode == "unity":
        # the authored joints and AddComponent's spares
        glue = _with_joints( glue, max( len( plan.get( "joints2d" ) or [] ),
                                        int( plan.get( "joints2d_cap" ) or 0 ) ) )
    if plan.get( "physics2d_polygons" ):
        glue = _with_polygons( glue, bool( plan.get( "physics2d_contacts" ) ) )
    if plan.get( "physics2d_terrain" ):
        if not plan.get( "physics2d_polygons" ):
            glue = _with_pair_refs( glue )  # a chunk is many shapes, as a polygon collider is
        chunks = sum( 1 for c in ( plan.get( "collider2d" ) or [] ) if c.get( "kind" ) == TERRAIN_KIND )
        glue = _with_terrain( glue, max( 1, chunks ), max( 1, int( plan.get( "terrain2d_max_shapes" ) or 256 ) ) )
        if plan.get( "terrain2d_chains" ):
            glue = _with_terrain_chains( glue, max( 1, int( plan.get( "terrain2d_max_chains" ) or 64 ) ),
                                         max( 2, int( plan.get( "terrain2d_max_points" ) or 1024 ) ) )
    paths = []
    glue_path = os.path.join( outdir, GLUE_FILE )
    _write_if_different( glue_path, glue )
    paths.append( glue_path )

    inject_path = os.path.join( outdir, INJECT_FILE )
    if inject:
        spec = {
            "comment": "Generated by box2d_unity.py for unity_pack --physics-inject. "
                       "Record touching pairs at the contact markers; scripts run after the step.",
            "defines": {
                "B2_PACK_NO_CONTACT_BEGIN_ARRAY": 1,
                "B2_PACK_NO_CONTACT_END_ARRAY": 1,
            },
            "globals": [
                "void b2u_on_begin( int colliderA, int colliderB );",
                "void b2u_on_end( int colliderA, int colliderB );",
            ],
            "inject": [
                {
                    "event": "contact_begin",
                    "code": "b2u_on_begin( (int)(intptr_t)shapeA->userData - 1, (int)(intptr_t)shapeB->userData - 1 );",
                },
                {
                    "event": "contact_end",
                    "code": "b2u_on_end( (int)(intptr_t)shapeA->userData - 1, (int)(intptr_t)shapeB->userData - 1 );",
                },
            ],
        }
        if plan.get( "physics2d_triggers" ) and mode == "unity":
            # OnTrigger*2D: the sensor events are injected too (_with_unity_triggers)
            spec["defines"]["B2_PACK_NO_SENSOR_BEGIN_ARRAY"] = 1
            spec["defines"]["B2_PACK_NO_SENSOR_END_ARRAY"] = 1
            spec["globals"] += [
                "void b2u_trig_begin( int a, int b );",
                "void b2u_trig_end( int a, int b );",
            ]
            spec["inject"] += [
                {
                    "event": "sensor_begin",
                    "code": "b2u_trig_begin( (int)(intptr_t)sensorShape->userData - 1, (int)(intptr_t)visitorShape->userData - 1 );",
                },
                {
                    "event": "sensor_end",
                    "code": "if ( visitorShape != NULL ) b2u_trig_end( (int)(intptr_t)sensorShape->userData - 1, (int)(intptr_t)visitorShape->userData - 1 );",
                },
            ]
        if mode == "godot":
            spec["defines"]["B2_PACK_NO_SENSOR_BEGIN_ARRAY"] = 1
            spec["defines"]["B2_PACK_NO_SENSOR_END_ARRAY"] = 1
            spec["inject"] += [
                {
                    "event": "sensor_begin",
                    "code": "b2u_on_begin( (int)(intptr_t)sensorShape->userData - 1, (int)(intptr_t)visitorShape->userData - 1 );",
                },
                {
                    "event": "sensor_end",
                    "code": "if ( visitorShape != NULL ) b2u_on_end( (int)(intptr_t)sensorShape->userData - 1, (int)(intptr_t)visitorShape->userData - 1 );",
                },
            ]
        _write_if_different( inject_path, json.dumps( spec, indent=2 ) + "\n" )
        paths.append( inject_path )
    elif os.path.exists( inject_path ):
        os.remove( inject_path )
    return paths


def build_library( outdir, inject=False, lto=False, box2d_root=None, verbose=False ):
    """
    Build Box2D-Packed for this pack with box2d_pack, into outdir/box2d.
    Returns (library path, include directory, extra compiler defines for the glue).
    """
    root = os.path.abspath( box2d_root or _HERE )
    if root not in sys.path:
        sys.path.insert( 0, root )
    import box2d_pack  # noqa: E402

    inject_path = os.path.join( outdir, INJECT_FILE ) if inject else None
    result = box2d_pack.build(
        None,
        inject_path,
        build_dir=os.path.join( outdir, "box2d" ),
        lto=lto,
        repo_root=root,
        verbose=verbose,
    )
    include_dir = os.path.join( result.source_dir, "include" )
    defines = ["-DB2_PACK_INJECTED=1"] if inject else []
    return result.lib, include_dir, defines


def _with_live_gate( glue ):
    """Glue that disables the bodies of GameObjects engine.c reports as not live."""
    edits = (
        ( "void engine_col2d_contact( int a, int b );\n",
          "void engine_col2d_contact( int a, int b );\n" + LIVE_EXPORTS ),
        ( "static int b2u_pair_n;\n", "static int b2u_pair_n;\n" + LIVE_STATE ),
        ( "\t\tb2BodyId bodyId = b2CreateBody( b2u_world, &def );\n"
          "\t\tb2u_add_shape( bodyId, ci, b2Vec2_zero );\n",
          "\t\tb2BodyId bodyId = b2CreateBody( b2u_world, &def );\n"
          "\t\tb2u_add_shape( bodyId, ci, b2Vec2_zero );\n"
          "\t\tb2u_col_on[ci] = 1;\n" ),
        ( "\t\tb2u_create_body( b2u_rb_created );\n",
          "\t\tb2u_create_body( b2u_rb_created );\n"
          "\t\tb2u_rb_on[b2u_rb_created] = 1;\n" ),
        ( "\t/* Push what scripts may have changed since the last step */\n",
          LIVE_SYNC + "\n\t/* Push what scripts may have changed since the last step */\n" ),
        ( "\t\tb2BodyId bodyId = b2u_rb_body[rb];\n\t\tfloat x, y;\n",
          "\t\tif ( b2u_rb_on[rb] == 0 )\n\t\t\tcontinue;\n"
          "\t\tb2BodyId bodyId = b2u_rb_body[rb];\n\t\tfloat x, y;\n" ),
        ( "\t\tb2BodyId bodyId = b2u_rb_body[rb];\n\t\tb2Pos p = b2Body_GetPosition( bodyId );\n",
          "\t\tif ( b2u_rb_on[rb] == 0 )\n\t\t\tcontinue;\n"
          "\t\tb2BodyId bodyId = b2u_rb_body[rb];\n\t\tb2Pos p = b2Body_GetPosition( bodyId );\n" ),
    )
    for old, new in edits:
        if glue.count( old ) != 1:
            raise ValueError( "box2d_unity: live gate anchor not found: %r" % old[:60] )
        glue = glue.replace( old, new )
    return glue


def _with_contact_manifolds( glue ):
    """Glue that reports each touching pair's manifold (normal, points) before the pair."""
    edits = (
        ( "void engine_col2d_contact( int a, int b );\n",
          "void engine_col2d_contact( int a, int b );\n" + CONTACT_EXPORTS ),
        ( "static int b2u_pair_n;\n", "static int b2u_pair_n;\n" + CONTACT_STATE ),
        ( "\t\tb2CreateCircleShape( bodyId, &def, &circle );\n",
          "\t\tb2u_col_shape[ci] = b2CreateCircleShape( bodyId, &def, &circle );\n"
          "\t\tb2u_col_has_shape[ci] = 1;\n" ),
        ( "\t\tb2CreateCapsuleShape( bodyId, &def, &capsule );\n",
          "\t\tb2u_col_shape[ci] = b2CreateCapsuleShape( bodyId, &def, &capsule );\n"
          "\t\tb2u_col_has_shape[ci] = 1;\n" ),
        ( "\t\tb2CreatePolygonShape( bodyId, &def, &box );\n",
          "\t\tb2u_col_shape[ci] = b2CreatePolygonShape( bodyId, &def, &box );\n"
          "\t\tb2u_col_has_shape[ci] = 1;\n" ),
        ( "\t\tengine_col2d_contact( b2u_pair_a[i], b2u_pair_b[i] );\n",
          "\t\tb2u_report_manifold( b2u_pair_a[i], b2u_pair_b[i] );\n"
          "\t\tengine_col2d_contact( b2u_pair_a[i], b2u_pair_b[i] );\n" ),
    )
    for old, new in edits:
        if glue.count( old ) != 1:
            raise ValueError( "box2d_unity: contact anchor not found: %r" % old[:60] )
        glue = glue.replace( old, new )
    return glue


#: Physics2D.Raycast / RaycastAll / OverlapCircle / OverlapCircleAll / OverlapPoint, with Unity's
#: layerMask, in every glue (unused ones cost nothing). Each returns the collider index hit (-1
#: for none), or how many. Triggers are hit, as Unity's queriesHitTriggers default has it. A ray
#: starting inside a collider hits it at its origin (distance 0, normal against the ray) while
#: engine_box2d_queries_start_in_colliders is set, Physics2D.queriesStartInColliders (default
#: true); cleared, the ray passes through that collider.
QUERY_FUNCTIONS = """
/* Physics2D queries for unity_pack. Unity's layerMask is tested against each collider's layer
 * (m_Layer) in the callbacks: Box2D-Packed's filters are 16 bits. */
#include <math.h>
int engine_box2d_queries_start_in_colliders = 1;
int engine_box2d_raycast( float ox, float oy, float dx, float dy, float distance, unsigned int mask, float* out );
int engine_box2d_raycast_all( float ox, float oy, float dx, float dy, float distance, unsigned int mask,
							  float* out, int* colliders, int max );
int engine_box2d_overlap_circle( float x, float y, float radius, unsigned int mask );
int engine_box2d_overlap_circle_all( float x, float y, float radius, unsigned int mask, int* colliders, int max );
int engine_box2d_overlap_point( float x, float y, unsigned int mask );

/* The collider of a shape, or -1 when the mask leaves its layer out */
static int b2u_query_collider( b2ShapeId shapeId, unsigned int mask )
{
	int ci = (int)(intptr_t)b2Shape_GetUserData( shapeId ) - 1;
	if ( ci < 0 || ( ( mask >> ( _Collider2D_layer[ci] & 31 ) ) & 1u ) == 0 )
		return -1;
	return ci;
}

/* The ray's translation, or 0 for no ray: its length is `distance` (capped) */
static int b2u_ray( float dx, float dy, float* distance, b2Vec2* translation )
{
	float len = sqrtf( dx * dx + dy * dy );
	if ( len <= 0.0f )
		return 0;
	if ( !( *distance < 1.0e6f ) )
		*distance = 1.0e6f;
	translation->x = dx / len * *distance;
	translation->y = dy / len * *distance;
	return 1;
}

typedef struct b2uRayAll
{
	float* out;
	int* colliders;
	int n, max, closest;
	float distance;
	unsigned int mask;
	const int* inside;
	int n_inside;
} b2uRayAll;

/* ponytail: the first 8 colliders holding a ray's origin; a 9th is cast through as an edge */
#define B2U_MAX_INSIDE 8

/* The colliders on the mask that hold the ray's origin: hit there, or passed through */
static int b2u_inside( float x, float y, unsigned int mask, int* inside )
{
	return engine_box2d_overlap_circle_all( x, y, 0.0f, mask, inside, B2U_MAX_INSIDE );
}

/* A hit at the origin of a ray that starts inside a collider: distance 0, normal against the ray */
static void b2u_start_hit( float ox, float oy, b2Vec2 translation, float* o )
{
	b2Vec2 n = b2Normalize( translation );
	o[0] = ox;
	o[1] = oy;
	o[2] = -n.x;
	o[3] = -n.y;
	o[4] = 0.0f;
	o[5] = 0.0f;
}

static float b2u_ray_hit( b2ShapeId shapeId, b2Pos point, b2Vec2 normal, float fraction, void* context )
{
	b2uRayAll* all = context;
	int ci = b2u_query_collider( shapeId, all->mask );
	for ( int i = 0; ci >= 0 && i < all->n_inside; ++i )
	{
		if ( all->inside[i] == ci )
			ci = -1; /* a polygon's inner edge, or a collider it started in */
	}
	if ( ci < 0 )
		return -1.0f; /* not on the mask: ignore it, go on */
	int k = all->closest ? 0 : all->n;
	if ( all->closest == 0 && all->n >= all->max )
		return 1.0f;
	float* o = all->out + 6 * k;
	o[0] = (float)point.x;
	o[1] = (float)point.y;
	o[2] = normal.x;
	o[3] = normal.y;
	o[4] = fraction;
	o[5] = fraction * all->distance;
	all->colliders[k] = ci;
	if ( all->closest )
	{
		all->n = 1;
		return fraction; /* clip: only a nearer one after this */
	}
	all->n += 1;
	return 1.0f; /* every shape on the ray */
}

/* out: point x, y, normal x, y, fraction, distance */
int engine_box2d_raycast( float ox, float oy, float dx, float dy, float distance, unsigned int mask, float* out )
{
	b2u_ensure();
	b2Vec2 translation;
	if ( b2u_ray( dx, dy, &distance, &translation ) == 0 )
		return -1;
	int inside[B2U_MAX_INSIDE];
	int n_inside = b2u_inside( ox, oy, mask, inside );
	if ( n_inside > 0 && engine_box2d_queries_start_in_colliders )
	{
		b2u_start_hit( ox, oy, translation, out );
		return inside[0];
	}
	int collider = -1;
	b2uRayAll one = { out, &collider, 0, 1, 1, distance, mask, inside, n_inside };
	b2World_CastRay( b2u_world, (b2Pos){ ox, oy }, translation, b2DefaultQueryFilter(), b2u_ray_hit, &one );
	return one.n > 0 ? collider : -1;
}

/* Every hit, nearest first (Unity's RaycastAll order); out holds 6 floats a hit */
int engine_box2d_raycast_all( float ox, float oy, float dx, float dy, float distance, unsigned int mask,
							  float* out, int* colliders, int max )
{
	b2u_ensure();
	b2Vec2 translation;
	if ( b2u_ray( dx, dy, &distance, &translation ) == 0 )
		return 0;
	int inside[B2U_MAX_INSIDE];
	int n_inside = b2u_inside( ox, oy, mask, inside );
	int n = 0;
	for ( ; engine_box2d_queries_start_in_colliders && n < n_inside && n < max; ++n )
	{
		b2u_start_hit( ox, oy, translation, out + 6 * n );
		colliders[n] = inside[n];
	}
	b2uRayAll all = { out, colliders, n, max, 0, distance, mask, inside, n_inside };
	b2World_CastRay( b2u_world, (b2Pos){ ox, oy }, translation, b2DefaultQueryFilter(), b2u_ray_hit, &all );
	for ( int i = 1; i < all.n; ++i )
	{
		for ( int k = i; k > 0 && out[6 * k + 4] < out[6 * ( k - 1 ) + 4]; --k )
		{
			for ( int j = 0; j < 6; ++j )
			{
				float t = out[6 * k + j];
				out[6 * k + j] = out[6 * ( k - 1 ) + j];
				out[6 * ( k - 1 ) + j] = t;
			}
			int c = colliders[k];
			colliders[k] = colliders[k - 1];
			colliders[k - 1] = c;
		}
	}
	return all.n;
}

typedef struct b2uOverlap
{
	int* colliders;
	int n, max;
	unsigned int mask;
} b2uOverlap;

static bool b2u_overlap_collect( b2ShapeId shapeId, void* context )
{
	b2uOverlap* o = context;
	int ci = b2u_query_collider( shapeId, o->mask );
	for ( int i = 0; ci >= 0 && i < o->n; ++i )
	{
		if ( o->colliders[i] == ci )
			ci = -1; /* another shape of a polygon collider already in */
	}
	if ( ci >= 0 && o->n < o->max )
	{
		o->colliders[o->n] = ci;
		o->n += 1;
	}
	return o->n < o->max;
}

int engine_box2d_overlap_circle_all( float x, float y, float radius, unsigned int mask, int* colliders, int max )
{
	b2u_ensure();
	b2Vec2 center = { 0.0f, 0.0f };
	b2ShapeProxy proxy = b2MakeProxy( &center, 1, radius > 0.0f ? radius : 0.0f );
	b2uOverlap o = { colliders, 0, max, mask };
	b2World_OverlapShape( b2u_world, (b2Pos){ x, y }, &proxy, b2DefaultQueryFilter(), b2u_overlap_collect, &o );
	return o.n;
}

int engine_box2d_overlap_circle( float x, float y, float radius, unsigned int mask )
{
	int found = -1;
	return engine_box2d_overlap_circle_all( x, y, radius, mask, &found, 1 ) > 0 ? found : -1;
}

int engine_box2d_overlap_point( float x, float y, unsigned int mask )
{
	return engine_box2d_overlap_circle( x, y, 0.0f, mask );
}
"""


JOINT_FUNCTIONS = r"""
/* 2D joints (plan["physics2d_joints"]): unity_pack's _Joint2D_* tables. The connected body is A
 * and the joint's own body B (a wheel: its own body, the chassis, A); no connected body is a
 * static ground body at the origin. Frames make the joint's angle 0 at creation, so a hinge's
 * limits and angle are relative to its pose then, as Unity's are. */
enum {{ B2U_MAX_JOINT = {N} }};
extern int _Joint2D_count;
extern int _Joint2D_kind[], _Joint2D_rb_a[], _Joint2D_rb_b[], _Joint2D_enabled[], _Joint2D_collide[];
extern int _Joint2D_auto_anchor[], _Joint2D_auto_distance[], _Joint2D_max_distance_only[];
extern int _Joint2D_use_motor[], _Joint2D_use_limits[], _Joint2D_auto_angle[], _Joint2D_broken[];
extern float _Joint2D_anchor_x[], _Joint2D_anchor_y[], _Joint2D_canchor_x[], _Joint2D_canchor_y[];
extern float _Joint2D_distance[], _Joint2D_frequency[], _Joint2D_damping[], _Joint2D_lower[];
extern float _Joint2D_upper[], _Joint2D_motor_speed[], _Joint2D_motor_max[], _Joint2D_angle[];
extern float _Joint2D_break_force[], _Joint2D_break_torque[];
extern float _Joint2D_out_angle[], _Joint2D_out_speed[], _Joint2D_out_translation[];
extern int _Joint2D_auto_offset[], _Joint2D_auto_target[], _Joint2D_break_action[];
extern float _Joint2D_max_force[], _Joint2D_max_torque[], _Joint2D_correction[];
extern float _Joint2D_offset_x[], _Joint2D_offset_y[], _Joint2D_offset_angle[];
extern float _Joint2D_target_x[], _Joint2D_target_y[];
extern float _Joint2D_out_force_x[], _Joint2D_out_force_y[], _Joint2D_out_torque[];
void engine_joint2d_broken( int j );

#define B2U_DEG ( B2_PI / 180.0f )

static b2JointId b2u_joint[B2U_MAX_JOINT];
static int b2u_joint_live[B2U_MAX_JOINT];
static int b2u_joints_ready;
static b2BodyId b2u_ground;
/* settings as last pushed: use_motor, motor_speed, motor_max, use_limits, lower, upper,
 * distance, frequency, damping, break_force, break_torque, max_force, max_torque, correction,
 * offset x, y, angle, target x, y, break_action; and what the joint was built from (a change
 * rebuilds it): connected body, anchor x, y, connected anchor x, y */
enum {{ B2U_JLAST = 25 }};
static float b2u_jlast[B2U_MAX_JOINT][B2U_JLAST];

static int b2u_jchanged( int j, int k, float v )
{{
	if ( b2u_jlast[j][k] == v )
		return 0;
	b2u_jlast[j][k] = v;
	return 1;
}}

/* A break threshold; JointBreakAction2D.Ignore (0) never breaks */
static float b2u_threshold( int j, float v )
{{
	if ( _Joint2D_break_action[j] == 0 )
		return 3.0e38f;
	return v < 1.0e37f && v >= 0.0f ? v : 3.0e38f;
}}

/* RelativeJoint2D's correctionScale (Box2D v2's motor joint corrected that fraction of the error
 * a step) as a spring: sqrt( scale ) / ( 2 pi dt ) hertz */
static float b2u_correction_hertz( int j )
{{
	float h = Time_fixedDeltaTime > 1e-8f ? Time_fixedDeltaTime : 0.02f;
	float c = _Joint2D_correction[j] > 0.0f ? _Joint2D_correction[j] : 0.0f;
	return sqrtf( c ) / ( 2.0f * B2_PI * h );
}}

static void b2u_joint_base( b2JointDef* base, int j, b2BodyId a, b2Vec2 pa, float qa, b2BodyId b, b2Vec2 pb,
							float qb )
{{
	base->bodyIdA = a;
	base->bodyIdB = b;
	base->localFrameA.p = pa;
	base->localFrameA.q = b2MakeRot( qa );
	base->localFrameB.p = pb;
	base->localFrameB.q = b2MakeRot( qb );
	base->collideConnected = _Joint2D_collide[j] != 0;
	base->forceThreshold = b2u_threshold( j, _Joint2D_break_force[j] );
	base->torqueThreshold = b2u_threshold( j, _Joint2D_break_torque[j] );
	base->userData = (void*)(intptr_t)( j + 1 );
}}

static void b2u_create_joint( int j )
{{
	int ra = _Joint2D_rb_a[j], rb = _Joint2D_rb_b[j];
	if ( ra < 0 || ra >= b2u_rb_created )
		return;
	b2BodyId own = b2u_rb_body[ra];
	b2BodyId other = rb >= 0 && rb < b2u_rb_created ? b2u_rb_body[rb] : b2u_ground;
	b2Vec2 la = {{ _Joint2D_anchor_x[j], _Joint2D_anchor_y[j] }};
	b2Pos wa = b2Body_GetWorldPoint( own, la );
	/* autoConfigureConnectedAnchor: where the own anchor is now, on the connected body */
	b2Vec2 lc = _Joint2D_auto_anchor[j] ? b2Body_GetLocalPoint( other, wa )
										: (b2Vec2){{ _Joint2D_canchor_x[j], _Joint2D_canchor_y[j] }};
	b2Pos wc = b2Body_GetWorldPoint( other, lc );
	_Joint2D_canchor_x[j] = lc.x;
	_Joint2D_canchor_y[j] = lc.y;
	float angOwn = b2Rot_GetAngle( b2Body_GetRotation( own ) );
	float angOther = b2Rot_GetAngle( b2Body_GetRotation( other ) );
	float dx = (float)( wa.x - wc.x ), dy = (float)( wa.y - wc.y );
	b2JointId id = b2_nullJointId;
	switch ( _Joint2D_kind[j] )
	{{
		case 0: /* HingeJoint2D */
		{{
			b2RevoluteJointDef def = b2DefaultRevoluteJointDef();
			b2u_joint_base( &def.base, j, other, lc, angOwn - angOther, own, la, 0.0f );
			def.enableLimit = _Joint2D_use_limits[j] != 0;
			def.lowerAngle = _Joint2D_lower[j] * B2U_DEG;
			def.upperAngle = _Joint2D_upper[j] * B2U_DEG;
			if ( def.lowerAngle > def.upperAngle )
				def.upperAngle = def.lowerAngle;
			def.enableMotor = _Joint2D_use_motor[j] != 0;
			def.motorSpeed = _Joint2D_motor_speed[j] * B2U_DEG;
			def.maxMotorTorque = _Joint2D_motor_max[j];
			id = b2CreateRevoluteJoint( b2u_world, &def );
			break;
		}}
		case 1: /* DistanceJoint2D */
		case 2: /* SpringJoint2D */
		{{
			b2DistanceJointDef def = b2DefaultDistanceJointDef();
			b2u_joint_base( &def.base, j, other, lc, 0.0f, own, la, 0.0f );
			if ( _Joint2D_auto_distance[j] )
				_Joint2D_distance[j] = sqrtf( dx * dx + dy * dy );
			float length = _Joint2D_distance[j] > 0.005f ? _Joint2D_distance[j] : 0.005f;
			def.length = length;
			def.minLength = 0.0f;
			def.maxLength = length;
			if ( _Joint2D_kind[j] == 2 )
			{{
				def.enableSpring = _Joint2D_frequency[j] > 0.0f;
				def.hertz = _Joint2D_frequency[j];
				def.dampingRatio = _Joint2D_damping[j];
			}}
			else if ( _Joint2D_max_distance_only[j] )
			{{
				/* a rope: slack up to the distance, taut at it */
				def.enableSpring = true;
				def.hertz = 0.0f;
				def.enableLimit = true;
			}}
			id = b2CreateDistanceJoint( b2u_world, &def );
			break;
		}}
		case 3: /* FixedJoint2D */
		{{
			b2WeldJointDef def = b2DefaultWeldJointDef();
			b2u_joint_base( &def.base, j, other, lc, angOwn - angOther, own, la, 0.0f );
			def.linearHertz = _Joint2D_frequency[j];
			def.angularHertz = _Joint2D_frequency[j];
			def.linearDampingRatio = _Joint2D_damping[j];
			def.angularDampingRatio = _Joint2D_damping[j];
			id = b2CreateWeldJoint( b2u_world, &def );
			break;
		}}
		case 4: /* SliderJoint2D: the axis in the world, degrees */
		{{
			if ( _Joint2D_auto_angle[j] && dx * dx + dy * dy > 1.0e-12f )
				_Joint2D_angle[j] = atan2f( dy, dx ) / B2U_DEG;
			float axis = _Joint2D_angle[j] * B2U_DEG;
			b2PrismaticJointDef def = b2DefaultPrismaticJointDef();
			b2u_joint_base( &def.base, j, other, lc, axis - angOther, own, la, axis - angOwn );
			def.enableLimit = _Joint2D_use_limits[j] != 0;
			def.lowerTranslation = _Joint2D_lower[j];
			def.upperTranslation = _Joint2D_upper[j] > _Joint2D_lower[j] ? _Joint2D_upper[j] : _Joint2D_lower[j];
			def.enableMotor = _Joint2D_use_motor[j] != 0;
			def.motorSpeed = _Joint2D_motor_speed[j];
			def.maxMotorForce = _Joint2D_motor_max[j];
			id = b2CreatePrismaticJoint( b2u_world, &def );
			break;
		}}
		case 5: /* WheelJoint2D: on the chassis, connected to the wheel */
		{{
			float axis = _Joint2D_angle[j] * B2U_DEG;
			b2WheelJointDef def = b2DefaultWheelJointDef();
			b2u_joint_base( &def.base, j, own, la, axis - angOwn, other, lc, axis - angOther );
			def.enableSpring = _Joint2D_frequency[j] > 0.0f;
			def.hertz = _Joint2D_frequency[j];
			def.dampingRatio = _Joint2D_damping[j];
			def.enableLimit = _Joint2D_use_limits[j] != 0;
			def.lowerTranslation = _Joint2D_lower[j];
			def.upperTranslation = _Joint2D_upper[j] > _Joint2D_lower[j] ? _Joint2D_upper[j] : _Joint2D_lower[j];
			def.enableMotor = _Joint2D_use_motor[j] != 0;
			def.motorSpeed = _Joint2D_motor_speed[j] * B2U_DEG;
			def.maxMotorTorque = _Joint2D_motor_max[j];
			id = b2CreateWheelJoint( b2u_world, &def );
			break;
		}}
		case 6: /* FrictionJoint2D: relative motion to rest, up to maxForce / maxTorque */
		{{
			b2MotorJointDef def = b2DefaultMotorJointDef();
			b2u_joint_base( &def.base, j, other, lc, angOwn - angOther, own, la, 0.0f );
			def.maxVelocityForce = _Joint2D_max_force[j];
			def.maxVelocityTorque = _Joint2D_max_torque[j];
			def.linearHertz = 0.0f;
			def.angularHertz = 0.0f;
			def.maxSpringForce = 0.0f;
			def.maxSpringTorque = 0.0f;
			id = b2CreateMotorJoint( b2u_world, &def );
			break;
		}}
		case 7: /* RelativeJoint2D: the own body held at an offset from the connected one */
		{{
			if ( _Joint2D_auto_offset[j] )
			{{
				b2Vec2 off = b2Body_GetLocalPoint( other, b2Body_GetPosition( own ) );
				_Joint2D_offset_x[j] = off.x;
				_Joint2D_offset_y[j] = off.y;
				_Joint2D_offset_angle[j] = ( angOwn - angOther ) / B2U_DEG;
			}}
			b2MotorJointDef def = b2DefaultMotorJointDef();
			b2u_joint_base( &def.base, j, other, (b2Vec2){{ _Joint2D_offset_x[j], _Joint2D_offset_y[j] }},
							_Joint2D_offset_angle[j] * B2U_DEG, own, (b2Vec2){{ 0.0f, 0.0f }}, 0.0f );
			def.maxVelocityForce = _Joint2D_max_force[j];
			def.maxVelocityTorque = _Joint2D_max_torque[j];
			def.linearHertz = b2u_correction_hertz( j );
			def.angularHertz = b2u_correction_hertz( j );
			def.linearDampingRatio = 1.0f;
			def.angularDampingRatio = 1.0f;
			def.maxSpringForce = _Joint2D_max_force[j];
			def.maxSpringTorque = _Joint2D_max_torque[j];
			id = b2CreateMotorJoint( b2u_world, &def );
			break;
		}}
		case 8: /* TargetJoint2D: the own anchor pulled to a world point by a spring */
		{{
			if ( _Joint2D_auto_target[j] )
			{{
				_Joint2D_target_x[j] = (float)wa.x;
				_Joint2D_target_y[j] = (float)wa.y;
			}}
			b2MotorJointDef def = b2DefaultMotorJointDef();
			b2u_joint_base( &def.base, j, b2u_ground, (b2Vec2){{ _Joint2D_target_x[j], _Joint2D_target_y[j] }},
							angOwn, own, la, 0.0f );
			def.maxVelocityForce = 0.0f;
			def.maxVelocityTorque = 0.0f;
			def.linearHertz = _Joint2D_frequency[j];
			def.linearDampingRatio = _Joint2D_damping[j];
			def.maxSpringForce = _Joint2D_max_force[j];
			def.angularHertz = 0.0f;
			def.maxSpringTorque = 0.0f;
			id = b2CreateMotorJoint( b2u_world, &def );
			break;
		}}
		default:
			break;
	}}
	b2u_joint[j] = id;
	b2u_joint_live[j] = B2_IS_NON_NULL( id );
	/* what was built is what was pushed */
	float now[B2U_JLAST] = {{ (float)_Joint2D_use_motor[j], _Joint2D_motor_speed[j], _Joint2D_motor_max[j],
							  (float)_Joint2D_use_limits[j], _Joint2D_lower[j], _Joint2D_upper[j],
							  _Joint2D_distance[j], _Joint2D_frequency[j], _Joint2D_damping[j],
							  _Joint2D_break_force[j], _Joint2D_break_torque[j], _Joint2D_max_force[j],
							  _Joint2D_max_torque[j], _Joint2D_correction[j], _Joint2D_offset_x[j],
							  _Joint2D_offset_y[j], _Joint2D_offset_angle[j], _Joint2D_target_x[j],
							  _Joint2D_target_y[j], (float)_Joint2D_break_action[j], (float)_Joint2D_rb_b[j],
							  _Joint2D_anchor_x[j], _Joint2D_anchor_y[j], _Joint2D_canchor_x[j],
							  _Joint2D_canchor_y[j] }};
	for ( int k = 0; k < B2U_JLAST; ++k )
		b2u_jlast[j][k] = now[k];
}}

static void b2u_create_joints( void )
{{
	b2BodyDef def = b2DefaultBodyDef();
	def.type = b2_staticBody;
	b2u_ground = b2CreateBody( b2u_world, &def );
	for ( int j = 0; j < _Joint2D_count && j < B2U_MAX_JOINT; ++j )
	{{
		if ( _Joint2D_enabled[j] && !_Joint2D_broken[j] )
			b2u_create_joint( j );
	}}
	b2u_joints_ready = 1;
}}

/* Before the step: a joint enabled or disabled, and each setting a script changed */
static void b2u_push_joints( void )
{{
	for ( int j = 0; j < _Joint2D_count && j < B2U_MAX_JOINT; ++j )
	{{
		if ( _Joint2D_broken[j] )
			continue;
		if ( b2u_joint_live[j] )
		{{
			/* connectedBody / anchor / connectedAnchor written by a script: built again */
			int moved = b2u_jchanged( j, 20, (float)_Joint2D_rb_b[j] ) | b2u_jchanged( j, 21, _Joint2D_anchor_x[j] ) |
						b2u_jchanged( j, 22, _Joint2D_anchor_y[j] ) | b2u_jchanged( j, 23, _Joint2D_canchor_x[j] ) |
						b2u_jchanged( j, 24, _Joint2D_canchor_y[j] );
			if ( moved )
			{{
				b2DestroyJoint( b2u_joint[j] );
				b2u_joint_live[j] = 0;
			}}
		}}
		if ( _Joint2D_enabled[j] && !b2u_joint_live[j] )
			b2u_create_joint( j );
		else if ( !_Joint2D_enabled[j] && b2u_joint_live[j] )
		{{
			b2DestroyJoint( b2u_joint[j] );
			b2u_joint_live[j] = 0;
		}}
		if ( !b2u_joint_live[j] )
			continue;
		b2JointId id = b2u_joint[j];
		int motorOn = b2u_jchanged( j, 0, (float)_Joint2D_use_motor[j] );
		int speed = b2u_jchanged( j, 1, _Joint2D_motor_speed[j] );
		int maxm = b2u_jchanged( j, 2, _Joint2D_motor_max[j] );
		int limOn = b2u_jchanged( j, 3, (float)_Joint2D_use_limits[j] );
		int lim = b2u_jchanged( j, 4, _Joint2D_lower[j] ) | b2u_jchanged( j, 5, _Joint2D_upper[j] );
		int dist = b2u_jchanged( j, 6, _Joint2D_distance[j] );
		int spring = b2u_jchanged( j, 7, _Joint2D_frequency[j] ) | b2u_jchanged( j, 8, _Joint2D_damping[j] );
		int action = b2u_jchanged( j, 19, (float)_Joint2D_break_action[j] );
		if ( b2u_jchanged( j, 9, _Joint2D_break_force[j] ) | action )
			b2Joint_SetForceThreshold( id, b2u_threshold( j, _Joint2D_break_force[j] ) );
		if ( b2u_jchanged( j, 10, _Joint2D_break_torque[j] ) | action )
			b2Joint_SetTorqueThreshold( id, b2u_threshold( j, _Joint2D_break_torque[j] ) );
		int maxf = b2u_jchanged( j, 11, _Joint2D_max_force[j] ) | b2u_jchanged( j, 12, _Joint2D_max_torque[j] );
		int corr = b2u_jchanged( j, 13, _Joint2D_correction[j] );
		int off = b2u_jchanged( j, 14, _Joint2D_offset_x[j] ) | b2u_jchanged( j, 15, _Joint2D_offset_y[j] ) |
				  b2u_jchanged( j, 16, _Joint2D_offset_angle[j] );
		int target = b2u_jchanged( j, 17, _Joint2D_target_x[j] ) | b2u_jchanged( j, 18, _Joint2D_target_y[j] );
		float lower = _Joint2D_lower[j];
		float upper = _Joint2D_upper[j] > lower ? _Joint2D_upper[j] : lower;
		switch ( _Joint2D_kind[j] )
		{{
			case 0:
				if ( motorOn )
					b2RevoluteJoint_EnableMotor( id, _Joint2D_use_motor[j] != 0 );
				if ( speed )
					b2RevoluteJoint_SetMotorSpeed( id, _Joint2D_motor_speed[j] * B2U_DEG );
				if ( maxm )
					b2RevoluteJoint_SetMaxMotorTorque( id, _Joint2D_motor_max[j] );
				if ( limOn )
					b2RevoluteJoint_EnableLimit( id, _Joint2D_use_limits[j] != 0 );
				if ( lim )
					b2RevoluteJoint_SetLimits( id, lower * B2U_DEG, upper * B2U_DEG );
				break;
			case 1:
			case 2:
				if ( dist )
				{{
					float length = _Joint2D_distance[j] > 0.005f ? _Joint2D_distance[j] : 0.005f;
					b2DistanceJoint_SetLength( id, length );
					b2DistanceJoint_SetLengthRange( id, 0.0f, length );
				}}
				if ( spring && _Joint2D_kind[j] == 2 )
				{{
					b2DistanceJoint_EnableSpring( id, _Joint2D_frequency[j] > 0.0f );
					b2DistanceJoint_SetSpringHertz( id, _Joint2D_frequency[j] );
					b2DistanceJoint_SetSpringDampingRatio( id, _Joint2D_damping[j] );
				}}
				break;
			case 3:
				if ( spring )
				{{
					b2WeldJoint_SetLinearHertz( id, _Joint2D_frequency[j] );
					b2WeldJoint_SetAngularHertz( id, _Joint2D_frequency[j] );
					b2WeldJoint_SetLinearDampingRatio( id, _Joint2D_damping[j] );
					b2WeldJoint_SetAngularDampingRatio( id, _Joint2D_damping[j] );
				}}
				break;
			case 4:
				if ( motorOn )
					b2PrismaticJoint_EnableMotor( id, _Joint2D_use_motor[j] != 0 );
				if ( speed )
					b2PrismaticJoint_SetMotorSpeed( id, _Joint2D_motor_speed[j] );
				if ( maxm )
					b2PrismaticJoint_SetMaxMotorForce( id, _Joint2D_motor_max[j] );
				if ( limOn )
					b2PrismaticJoint_EnableLimit( id, _Joint2D_use_limits[j] != 0 );
				if ( lim )
					b2PrismaticJoint_SetLimits( id, lower, upper );
				break;
			case 5:
				if ( motorOn )
					b2WheelJoint_EnableMotor( id, _Joint2D_use_motor[j] != 0 );
				if ( speed )
					b2WheelJoint_SetMotorSpeed( id, _Joint2D_motor_speed[j] * B2U_DEG );
				if ( maxm )
					b2WheelJoint_SetMaxMotorTorque( id, _Joint2D_motor_max[j] );
				if ( spring )
				{{
					b2WheelJoint_EnableSpring( id, _Joint2D_frequency[j] > 0.0f );
					b2WheelJoint_SetSpringHertz( id, _Joint2D_frequency[j] );
					b2WheelJoint_SetSpringDampingRatio( id, _Joint2D_damping[j] );
				}}
				if ( limOn )
					b2WheelJoint_EnableLimit( id, _Joint2D_use_limits[j] != 0 );
				if ( lim )
					b2WheelJoint_SetLimits( id, lower, upper );
				break;
			case 6:
			case 7:
				if ( maxf )
				{{
					b2MotorJoint_SetMaxVelocityForce( id, _Joint2D_max_force[j] );
					b2MotorJoint_SetMaxVelocityTorque( id, _Joint2D_max_torque[j] );
					if ( _Joint2D_kind[j] == 7 )
					{{
						b2MotorJoint_SetMaxSpringForce( id, _Joint2D_max_force[j] );
						b2MotorJoint_SetMaxSpringTorque( id, _Joint2D_max_torque[j] );
					}}
				}}
				if ( corr && _Joint2D_kind[j] == 7 )
				{{
					b2MotorJoint_SetLinearHertz( id, b2u_correction_hertz( j ) );
					b2MotorJoint_SetAngularHertz( id, b2u_correction_hertz( j ) );
				}}
				if ( off && _Joint2D_kind[j] == 7 )
				{{
					b2Transform frame = {{ {{ _Joint2D_offset_x[j], _Joint2D_offset_y[j] }},
										   b2MakeRot( _Joint2D_offset_angle[j] * B2U_DEG ) }};
					b2Joint_SetLocalFrameA( id, frame );
					b2Joint_WakeBodies( id );
				}}
				break;
			case 8:
				if ( maxf )
					b2MotorJoint_SetMaxSpringForce( id, _Joint2D_max_force[j] );
				if ( spring )
				{{
					b2MotorJoint_SetLinearHertz( id, _Joint2D_frequency[j] );
					b2MotorJoint_SetLinearDampingRatio( id, _Joint2D_damping[j] );
				}}
				if ( target )
				{{
					/* the ground's frame is the target (its rotation kept: the body turns freely) */
					b2Transform frame = b2Joint_GetLocalFrameA( id );
					frame.p = (b2Vec2){{ _Joint2D_target_x[j], _Joint2D_target_y[j] }};
					b2Joint_SetLocalFrameA( id, frame );
					b2Joint_WakeBodies( id );
				}}
				break;
			default:
				break;
		}}
	}}
}}

/* After the step: each joint's angle / speed / translation, and the joints that broke */
static void b2u_pull_joints( void )
{{
	for ( int j = 0; j < _Joint2D_count && j < B2U_MAX_JOINT; ++j )
	{{
		if ( !b2u_joint_live[j] )
			continue;
		b2JointId id = b2u_joint[j];
		b2Vec2 force = b2Joint_GetConstraintForce( id );
		_Joint2D_out_force_x[j] = force.x;
		_Joint2D_out_force_y[j] = force.y;
		_Joint2D_out_torque[j] = b2Joint_GetConstraintTorque( id );
		float wa = b2Body_GetAngularVelocity( b2Joint_GetBodyA( id ) );
		float wb = b2Body_GetAngularVelocity( b2Joint_GetBodyB( id ) );
		switch ( _Joint2D_kind[j] )
		{{
			case 0:
				_Joint2D_out_angle[j] = b2RevoluteJoint_GetAngle( id ) / B2U_DEG;
				_Joint2D_out_speed[j] = ( wb - wa ) / B2U_DEG;
				break;
			case 4:
				_Joint2D_out_translation[j] = b2PrismaticJoint_GetTranslation( id );
				_Joint2D_out_speed[j] = b2PrismaticJoint_GetSpeed( id );
				break;
			case 5:
				_Joint2D_out_speed[j] = ( wb - wa ) / B2U_DEG;
				break;
			default:
				break;
		}}
	}}
	/* breakForce / breakTorque: Box2D reports the joints past their thresholds. The event data
	 * goes stale once a joint is destroyed, so the indices are taken first. */
	b2JointEvents events = b2World_GetJointEvents( b2u_world );
	int broken[B2U_MAX_JOINT];
	int n = 0;
	for ( int i = 0; i < events.count && n < B2U_MAX_JOINT; ++i )
	{{
		int j = (int)(intptr_t)events.jointEvents[i].userData - 1;
		if ( j >= 0 && j < B2U_MAX_JOINT && b2u_joint_live[j] )
			broken[n++] = j;
	}}
	for ( int i = 0; i < n; ++i )
	{{
		int j = broken[i];
		if ( !b2u_joint_live[j] )
			continue;
		/* JointBreakAction2D: CallbackOnly (1) keeps the joint; Disable (2) and Destroy (3)
		 * remove it -- a disabled one comes back when a script enables it */
		if ( _Joint2D_break_action[j] >= 2 )
		{{
			b2DestroyJoint( b2u_joint[j] );
			b2u_joint_live[j] = 0;
		}}
		engine_joint2d_broken( j );
	}}
}}
"""


def _with_joints( glue, n ):
    """
    2D joints (plan["physics2d_joints"], unity mode): HingeJoint2D, DistanceJoint2D, SpringJoint2D,
    FixedJoint2D, SliderJoint2D, WheelJoint2D, and FrictionJoint2D / RelativeJoint2D /
    TargetJoint2D as motor joints, from unity_pack's _Joint2D_* tables. They are built
    with the bodies (b2u_ensure: a script's Start sees them), each step pushes what scripts changed
    (enabled, motor, limits, distance, spring, forces, offsets, target, break thresholds and action)
    and pulls the joint angle, speed, translation and reaction force / torque back; a joint past
    its breakForce / breakTorque is reported with engine_joint2d_broken( j ) and, by its
    JointBreakAction2D, kept (CallbackOnly), or removed (Disable, Destroy).
    """
    functions = JOINT_FUNCTIONS.format( N=max( 1, int( n ) ) )
    ensure_end = "\n}\n\nvoid engine_box2d_step( void )\n{"
    assert ensure_end in glue, "box2d_unity: joints marker (ensure) not found"
    # the functions go before b2u_ensure, which builds the joints with the bodies
    ensure_start = "static void b2u_ensure( void )\n{"
    assert ensure_start in glue, "box2d_unity: joints marker (ensure start) not found"
    glue = glue.replace( ensure_start, functions + "\n" + ensure_start, 1 )
    glue = glue.replace( ensure_end,
                         "\n\tif ( b2u_joints_ready == 0 && b2u_rb_created >= _Rigidbody2D_count )\n"
                         "\t\tb2u_create_joints();" + ensure_end, 1 )
    step = "\tfloat dt = Time_fixedDeltaTime > 1e-8f ? Time_fixedDeltaTime : "
    assert glue.count( step ) == 1, "box2d_unity: joints marker (step) not found once"
    glue = glue.replace( step, "\tb2u_push_joints();\n\n" + step, 1 )
    pulled = "\t/* unity_pack sends Enter / Stay / Exit by comparing with the previous step */\n"
    assert pulled in glue, "box2d_unity: joints marker (pull) not found"
    glue = glue.replace( pulled, "\tb2u_pull_joints();\n\n" + pulled, 1 )
    if "#include <math.h>" not in glue:
        glue = glue.replace( "#include <stdint.h>\n", "#include <stdint.h>\n#include <math.h>\n", 1 )
    return glue


def _with_rotation( glue ):
    """
    Rigidbody2D rotation (plan["physics2d_rotation"], unity mode). A body turns unless its
    Rigidbody2D freezes rotation (RigidbodyConstraints2D.FreezeRotation); it starts at its owner's
    authored angle and angular velocity. Each step pushes what scripts changed -- an angle written
    (rotation / MoveRotation), freezeRotation, angularVelocity, and the torque and angular impulse
    AddTorque accumulated (applied, then cleared) -- and a teleport keeps the rotation; after the
    step the angle and angular velocity are pulled back. engine.c exports

        float engine_rb2d_get_rot( int rb );          owner's rotation about z, radians
        void engine_rb2d_set_rot( int rb, float a );

    Without the flag every body's rotation stays locked, as before.
    """
    def sub( old, new ):
        nonlocal glue
        assert old in glue, "box2d_unity: rotation marker not found: %r" % old[:60]
        glue = glue.replace( old, new, 1 )

    sub( "void engine_box2d_step( void );\n",
         "void engine_box2d_step( void );\n"
         "float engine_rb2d_get_rot( int rb );\n"
         "void engine_rb2d_set_rot( int rb, float a );\n"
         "extern int _Rigidbody2D_freeze_rot[];\n"
         "extern float _Rigidbody2D_ang_vel[];\n"
         "extern float _Rigidbody2D_torque[];\n"
         "extern float _Rigidbody2D_ang_imp[];\n" )
    sub( "static int b2u_last_type[B2U_MAX_RB];\n",
         "static int b2u_last_type[B2U_MAX_RB];\n"
         "/* the angle as last pulled, and freezeRotation as last pushed */\n"
         "static float b2u_last_a[B2U_MAX_RB];\n"
         "static int b2u_last_freeze[B2U_MAX_RB];\n" )
    sub( "\tdef.motionLocks.angularZ = true;\n",
         "\tdef.motionLocks.angularZ = _Rigidbody2D_freeze_rot[rb] != 0;\n"
         "\tdef.rotation = b2MakeRot( engine_rb2d_get_rot( rb ) );\n"
         "\tdef.angularVelocity = _Rigidbody2D_ang_vel[rb];\n" )
    sub( "\tb2u_last_type[rb] = _Rigidbody2D_body_type[rb];\n}\n",
         "\tb2u_last_type[rb] = _Rigidbody2D_body_type[rb];\n"
         "\tb2u_last_a[rb] = engine_rb2d_get_rot( rb );\n"
         "\tb2u_last_freeze[rb] = _Rigidbody2D_freeze_rot[rb];\n}\n" )
    sub( "b2Body_SetTransform( bodyId, (b2Pos){ x, y }, b2Rot_identity );",
         "b2Body_SetTransform( bodyId, (b2Pos){ x, y }, b2Body_GetRotation( bodyId ) );" )
    sub( "\t\tif ( _Rigidbody2D_body_type[rb] != b2u_last_type[rb] )\n",
         "\t\t{\n"
         "\t\t\t/* rotation / MoveRotation written by a script */\n"
         "\t\t\tfloat a = engine_rb2d_get_rot( rb );\n"
         "\t\t\tif ( a != b2u_last_a[rb] )\n"
         "\t\t\t\tb2Body_SetTransform( bodyId, b2Body_GetPosition( bodyId ), b2MakeRot( a ) );\n"
         "\t\t}\n"
         "\t\tif ( _Rigidbody2D_freeze_rot[rb] != b2u_last_freeze[rb] )\n"
         "\t\t{\n"
         "\t\t\tb2MotionLocks locks = { 0 };\n"
         "\t\t\tlocks.angularZ = _Rigidbody2D_freeze_rot[rb] != 0;\n"
         "\t\t\tb2Body_SetMotionLocks( bodyId, locks );\n"
         "\t\t\tb2u_last_freeze[rb] = _Rigidbody2D_freeze_rot[rb];\n"
         "\t\t}\n"
         "\t\tif ( _Rigidbody2D_body_type[rb] != b2u_last_type[rb] )\n" )
    sub( "\t\t\tb2Body_SetLinearVelocity( bodyId, (b2Vec2){ _Rigidbody2D_vel_x[rb], _Rigidbody2D_vel_y[rb] } );\n",
         "\t\t\tb2Body_SetLinearVelocity( bodyId, (b2Vec2){ _Rigidbody2D_vel_x[rb], _Rigidbody2D_vel_y[rb] } );\n"
         "\t\t\tb2Body_SetAngularVelocity( bodyId, _Rigidbody2D_ang_vel[rb] );\n"
         "\t\t\tif ( _Rigidbody2D_torque[rb] != 0.0f )\n"
         "\t\t\t\tb2Body_ApplyTorque( bodyId, _Rigidbody2D_torque[rb], true );\n"
         "\t\t\tif ( _Rigidbody2D_ang_imp[rb] != 0.0f )\n"
         "\t\t\t\tb2Body_ApplyAngularImpulse( bodyId, _Rigidbody2D_ang_imp[rb], true );\n"
         "\t\t\t_Rigidbody2D_torque[rb] = 0.0f;\n"
         "\t\t\t_Rigidbody2D_ang_imp[rb] = 0.0f;\n" )
    sub( "\t\t_Rigidbody2D_vel_y[rb] = v.y;\n",
         "\t\t_Rigidbody2D_vel_y[rb] = v.y;\n"
         "\t\tengine_rb2d_set_rot( rb, b2Rot_GetAngle( b2Body_GetRotation( bodyId ) ) );\n"
         "\t\tb2u_last_a[rb] = engine_rb2d_get_rot( rb );\n"
         "\t\t_Rigidbody2D_ang_vel[rb] = b2Body_GetAngularVelocity( bodyId );\n" )
    return glue


def _with_unity_triggers( glue ):
    """
    OnTriggerEnter2D / Stay2D / Exit2D (plan["physics2d_triggers"], unity mode). Every shape takes
    sensor events -- a trigger collider is a sensor already -- and the overlapping sensor pairs are
    kept from Box2D's sensor begin and end events, apart from the touching pairs, and reported after
    the step with

        void engine_col2d_trigger( int a, int b );

    unity_pack sends the messages by comparing them with the step before, as it does collisions.
    With --physics-inject the sensor begin / end events are injected (b2u_trig_begin / _end),
    as the contacts are.
    """
    glue = glue.replace(
        "\tdef.enableContactEvents = _Collider2D_is_trigger[ci] == 0;\n",
        "\tdef.enableContactEvents = _Collider2D_is_trigger[ci] == 0;\n"
        "\t/* OnTrigger*2D: every shape takes sensor events */\n"
        "\tdef.enableSensorEvents = true;\n", 1 )
    glue = glue.replace(
        "void b2u_on_begin( int colliderA, int colliderB );\n",
        "void b2u_on_begin( int colliderA, int colliderB );\n"
        "void engine_col2d_trigger( int a, int b );\n", 1 )
    storage = (
        "/* Overlapping sensor pairs, lo < hi, for OnTrigger*2D */\n"
        "static int b2u_trig_a[B2U_MAX_PAIRS];\n"
        "static int b2u_trig_b[B2U_MAX_PAIRS];\n"
        "static int b2u_trig_n;\n\n"
        "void b2u_trig_begin( int a, int b );\n"
        "void b2u_trig_end( int a, int b );\n\n"
        "void b2u_trig_begin( int a, int b )\n"
        "{\n"
        "\tint lo = a < b ? a : b;\n"
        "\tint hi = a < b ? b : a;\n"
        "\tif ( lo < 0 || lo == hi )\n"
        "\t\treturn;\n"
        "\tfor ( int i = 0; i < b2u_trig_n; ++i )\n"
        "\t\tif ( b2u_trig_a[i] == lo && b2u_trig_b[i] == hi )\n"
        "\t\t\treturn;\n"
        "\tif ( b2u_trig_n < B2U_MAX_PAIRS )\n"
        "\t{\n"
        "\t\tb2u_trig_a[b2u_trig_n] = lo;\n"
        "\t\tb2u_trig_b[b2u_trig_n] = hi;\n"
        "\t\tb2u_trig_n += 1;\n"
        "\t}\n"
        "}\n\n"
        "void b2u_trig_end( int a, int b )\n"
        "{\n"
        "\tint lo = a < b ? a : b;\n"
        "\tint hi = a < b ? b : a;\n"
        "\tfor ( int i = 0; i < b2u_trig_n; ++i )\n"
        "\t{\n"
        "\t\tif ( b2u_trig_a[i] == lo && b2u_trig_b[i] == hi )\n"
        "\t\t{\n"
        "\t\t\tfor ( int k = i + 1; k < b2u_trig_n; ++k )\n"
        "\t\t\t{\n"
        "\t\t\t\tb2u_trig_a[k - 1] = b2u_trig_a[k];\n"
        "\t\t\t\tb2u_trig_b[k - 1] = b2u_trig_b[k];\n"
        "\t\t\t}\n"
        "\t\t\tb2u_trig_n -= 1;\n"
        "\t\t\treturn;\n"
        "\t\t}\n"
        "\t}\n"
        "}\n\n" )
    glue = glue.replace( "void b2u_on_begin( int colliderA, int colliderB )\n{",
                         storage + "void b2u_on_begin( int colliderA, int colliderB )\n{", 1 )
    events = (
        "\t/* OnTrigger*2D: sensor overlaps, apart from the touching pairs */\n"
        "\tb2SensorEvents sensors = b2World_GetSensorEvents( b2u_world );\n"
        "\tfor ( int i = 0; i < sensors.beginCount; ++i )\n"
        "\t{\n"
        "\t\tb2SensorBeginTouchEvent* e = sensors.beginEvents + i;\n"
        "\t\tb2u_trig_begin( (int)(intptr_t)b2Shape_GetUserData( e->sensorShapeId ) - 1,\n"
        "\t\t\t\t\t\t(int)(intptr_t)b2Shape_GetUserData( e->visitorShapeId ) - 1 );\n"
        "\t}\n"
        "\tfor ( int i = 0; i < sensors.endCount; ++i )\n"
        "\t{\n"
        "\t\tb2SensorEndTouchEvent* e = sensors.endEvents + i;\n"
        "\t\tif ( b2Shape_IsValid( e->sensorShapeId ) && b2Shape_IsValid( e->visitorShapeId ) )\n"
        "\t\t{\n"
        "\t\t\tb2u_trig_end( (int)(intptr_t)b2Shape_GetUserData( e->sensorShapeId ) - 1,\n"
        "\t\t\t\t\t\t  (int)(intptr_t)b2Shape_GetUserData( e->visitorShapeId ) - 1 );\n"
        "\t\t}\n"
        "\t}\n" )
    marker = "#endif\n\n\t/* Pull positions and velocities into the packed tables */"
    assert marker in glue, "box2d_unity: trigger events marker not found"
    glue = glue.replace( marker, events + marker, 1 )
    report_marker = "\t/* unity_pack sends Enter / Stay / Exit by comparing with the previous step */\n"
    assert report_marker in glue, "box2d_unity: trigger report marker not found"
    glue = glue.replace(
        report_marker,
        "\tfor ( int i = 0; i < b2u_trig_n; ++i )\n"
        "\t{\n"
        "\t\tengine_col2d_trigger( b2u_trig_a[i], b2u_trig_b[i] );\n"
        "\t}\n" + report_marker, 1 )
    return glue


POLYGON_SHAPES = """/* PolygonCollider2D (kind 4): its paths as the triangles unity_pack cut them into, in the
 * collider's frame about its center; one convex polygon shape each, all the collider's */
static void b2u_add_polygon( b2BodyId bodyId, const b2ShapeDef* def, int ci, b2Vec2 offset )
{
	b2Rot rotation = { _Collider2D_cos[ci], _Collider2D_sin[ci] };
	int t0 = _Collider2D_tri_start[ci];
	for ( int t = t0; t < t0 + _Collider2D_tri_count[ci]; ++t )
	{
		b2Vec2 p[3];
		for ( int k = 0; k < 3; ++k )
		{
			b2Vec2 v = { _Collider2D_tri_xy[6 * t + 2 * k], _Collider2D_tri_xy[6 * t + 2 * k + 1] };
			p[k] = b2Add( offset, b2RotateVector( rotation, v ) );
		}
		b2Hull hull = b2ComputeHull( p, 3 );
		if ( hull.count == 0 )
			continue; /* a sliver thinner than Box2D's linear slop */
		b2Polygon tri = b2MakePolygon( &hull, 0.0f );
		b2ShapeId shapeId = b2CreatePolygonShape( bodyId, def, &tri );
{KEEP_SHAPE}		(void)shapeId;
	}
}

"""

#: With manifolds, a polygon collider's pair reads its first triangle's contact.
POLYGON_KEEP_SHAPE = """\t\tif ( b2u_col_has_shape[ci] == 0 )
\t\t{
\t\t\tb2u_col_shape[ci] = shapeId;
\t\t\tb2u_col_has_shape[ci] = 1;
\t\t}
"""


def _with_pair_refs( glue ):
    """
    Count the shape pairs that touch in each collider pair, so a pair ends when its last shape pair
    does: for colliders of several shapes (PolygonCollider2D triangles, terrain chunks).
    """
    edits = [
        ( "static int b2u_pair_n;\n",
          "static int b2u_pair_n;\n"
          "/* How many shape pairs touch in each pair: a polygon collider is several shapes */\n"
          "static int b2u_pair_refs[B2U_MAX_PAIRS];\n" ),
        ( "\t\tif ( b2u_pair_a[i] == lo && b2u_pair_b[i] == hi )\n\t\t{\n\t\t\treturn;\n",
          "\t\tif ( b2u_pair_a[i] == lo && b2u_pair_b[i] == hi )\n\t\t{\n"
          "\t\t\tb2u_pair_refs[i] += 1;\n\t\t\treturn;\n" ),
        ( "\t\tb2u_pair_b[b2u_pair_n] = hi;\n",
          "\t\tb2u_pair_b[b2u_pair_n] = hi;\n\t\tb2u_pair_refs[b2u_pair_n] = 1;\n" ),
        ( "\t\t\t/* Keep order stable, messages are sent in pair order */\n",
          "\t\t\tb2u_pair_refs[i] -= 1;\n\t\t\tif ( b2u_pair_refs[i] > 0 )\n\t\t\t\treturn;\n"
          "\t\t\t/* Keep order stable, messages are sent in pair order */\n" ),
        ( "\t\t\t\tb2u_pair_b[k - 1] = b2u_pair_b[k];\n",
          "\t\t\t\tb2u_pair_b[k - 1] = b2u_pair_b[k];\n"
          "\t\t\t\tb2u_pair_refs[k - 1] = b2u_pair_refs[k];\n" ),
    ]
    # the live gate's forgetting and the trigger pairs, when the glue has them
    optional = [
        ( "\t\tb2u_pair_b[k] = b;\n\t\tk += 1;\n",
          "\t\tb2u_pair_b[k] = b;\n\t\tb2u_pair_refs[k] = b2u_pair_refs[i];\n\t\tk += 1;\n" ),
        ( "static int b2u_trig_n;\n",
          "static int b2u_trig_n;\nstatic int b2u_trig_refs[B2U_MAX_PAIRS];\n" ),
        ( "\t\tif ( b2u_trig_a[i] == lo && b2u_trig_b[i] == hi )\n\t\t\treturn;\n",
          "\t\tif ( b2u_trig_a[i] == lo && b2u_trig_b[i] == hi )\n\t\t{\n"
          "\t\t\tb2u_trig_refs[i] += 1;\n\t\t\treturn;\n\t\t}\n" ),
        ( "\t\tb2u_trig_b[b2u_trig_n] = hi;\n",
          "\t\tb2u_trig_b[b2u_trig_n] = hi;\n\t\tb2u_trig_refs[b2u_trig_n] = 1;\n" ),
        ( "\t\t\tfor ( int k = i + 1; k < b2u_trig_n; ++k )\n",
          "\t\t\tb2u_trig_refs[i] -= 1;\n\t\t\tif ( b2u_trig_refs[i] > 0 )\n\t\t\t\treturn;\n"
          "\t\t\tfor ( int k = i + 1; k < b2u_trig_n; ++k )\n" ),
        ( "\t\t\t\tb2u_trig_b[k - 1] = b2u_trig_b[k];\n",
          "\t\t\t\tb2u_trig_b[k - 1] = b2u_trig_b[k];\n"
          "\t\t\t\tb2u_trig_refs[k - 1] = b2u_trig_refs[k];\n" ),
    ]
    for old, new in edits:
        if glue.count( old ) != 1:
            raise ValueError( "box2d_unity: pair-count anchor not found: %r" % old[:60] )
        glue = glue.replace( old, new )
    for old, new in optional:
        if glue.count( old ) > 1:
            raise ValueError( "box2d_unity: pair-count anchor not unique: %r" % old[:60] )
        glue = glue.replace( old, new )
    return glue


def _with_polygons( glue, contacts=False ):
    """
    PolygonCollider2D (plan["physics2d_polygons"]): collider kind 4, whose triangles are in

        extern const int _Collider2D_tri_start[];   first triangle of each collider
        extern const int _Collider2D_tri_count[];   how many (0 for other kinds)
        extern const float _Collider2D_tri_xy[];    x0 y0 x1 y1 x2 y2 per triangle

    Box2D's polygons are convex and of at most 8 vertices, so a collider is several shapes, and a
    pair of colliders touches through several shape pairs: the touching and overlapping pairs are
    counted, and a pair ends when its last shape pair does.
    """
    glue = _with_pair_refs( glue )
    edits = [
        ( "extern const int _Collider2D_bounce_combine[];\n",
          "extern const int _Collider2D_bounce_combine[];\n"
          "extern const int _Collider2D_tri_start[];\n"
          "extern const int _Collider2D_tri_count[];\n"
          "extern const float _Collider2D_tri_xy[];\n" ),
        ( "static void b2u_add_shape( b2BodyId bodyId, int ci, b2Vec2 offset )\n",
          POLYGON_SHAPES.replace( "{KEEP_SHAPE}", POLYGON_KEEP_SHAPE if contacts else "" )
          + "static void b2u_add_shape( b2BodyId bodyId, int ci, b2Vec2 offset )\n" ),
        ( "\tb2Rot rotation = { _Collider2D_cos[ci], _Collider2D_sin[ci] };\n\t/* CapsuleCollider2D",
          "\tif ( _Collider2D_kind[ci] == 4 )\n\t{\n"
          "\t\tb2u_add_polygon( bodyId, &def, ci, offset );\n\t\treturn;\n\t}\n"
          "\tb2Rot rotation = { _Collider2D_cos[ci], _Collider2D_sin[ci] };\n\t/* CapsuleCollider2D" ),
    ]
    for old, new in edits:
        if glue.count( old ) != 1:
            raise ValueError( "box2d_unity: polygon anchor not found: %r" % old[:60] )
        glue = glue.replace( old, new )
    return glue


TERRAIN_KIND = 5

TERRAIN_SHAPES = """
/* Terrain chunks (collider kind 5): one static body each, whose box shapes the game replaces
 * with b2u_terrain_set whenever the terrain changes. Boxes that did not change keep their shape
 * (and so their touching pairs); only the ones that did are destroyed and created. */
#define B2U_TERRAIN_CHUNKS {CHUNKS}
#define B2U_TERRAIN_SHAPES {SHAPES}
static int b2u_terrain_slot[B2U_MAX_COL]; /* slot + 1; 0 for a collider that is not a terrain chunk */
static int b2u_terrain_used;
static int b2u_terrain_count[B2U_TERRAIN_CHUNKS];
static float b2u_terrain_box[B2U_TERRAIN_CHUNKS][B2U_TERRAIN_SHAPES][4];
static b2ShapeId b2u_terrain_shape[B2U_TERRAIN_CHUNKS][B2U_TERRAIN_SHAPES];

static void b2u_terrain_claim( int ci )
{
\tif ( b2u_terrain_used < B2U_TERRAIN_CHUNKS )
\t{
\t\tb2u_terrain_used += 1;
\t\tb2u_terrain_slot[ci] = b2u_terrain_used;
\t}
}

"""

TERRAIN_SET = """
/* Replace the shapes of terrain chunk ci with n boxes of 4 floats: center x, center y, half
 * width, half height, relative to the chunk's position. More than B2U_TERRAIN_SHAPES are cut. */
void b2u_terrain_set( int ci, const float* boxes, int n )
{
\tb2u_ensure();
\tif ( ci < 0 || ci >= _Collider2D_count || ci >= B2U_MAX_COL || b2u_terrain_slot[ci] == 0 || b2u_col_has_body[ci] == 0 )
\t{
\t\treturn;
\t}
\tif ( n > B2U_TERRAIN_SHAPES )
\t{
\t\tn = B2U_TERRAIN_SHAPES;
\t}
\tint s = b2u_terrain_slot[ci] - 1;
\tint old = b2u_terrain_count[s];
\tb2BodyId bodyId = b2u_col_body[ci];

\tstatic unsigned char kept[B2U_TERRAIN_SHAPES];
\tstatic float nbox[B2U_TERRAIN_SHAPES][4];
\tstatic b2ShapeId nshape[B2U_TERRAIN_SHAPES];
\tfor ( int j = 0; j < old; ++j )
\t{
\t\tkept[j] = 0;
\t}

\tb2ShapeDef def = b2DefaultShapeDef();
\tdef.userData = (void*)(intptr_t)( ci + 1 );
\tdef.material.friction = _Collider2D_friction[ci];
\tdef.material.restitution = _Collider2D_bounciness[ci];
\tdef.material.userMaterialId =
\t\t(uint64_t)( _Collider2D_friction_combine[ci] & 0xff ) | ( (uint64_t)( _Collider2D_bounce_combine[ci] & 0xff ) << 8 );
\tdef.isSensor = _Collider2D_is_trigger[ci] != 0;
\tdef.enableContactEvents = _Collider2D_is_trigger[ci] == 0;

\tfor ( int i = 0; i < n; ++i )
\t{
\t\tconst float* b = boxes + 4 * i;
\t\tint match = -1;
\t\t/* the same slot first: the boxes before a change keep their places */
\t\tfor ( int k = -1; k < old && match < 0; ++k )
\t\t{
\t\t\tint j = k < 0 ? i : k;
\t\t\tif ( j >= old || kept[j] )
\t\t\t{
\t\t\t\tcontinue;
\t\t\t}
\t\t\tconst float* o = b2u_terrain_box[s][j];
\t\t\tif ( o[0] == b[0] && o[1] == b[1] && o[2] == b[2] && o[3] == b[3] )
\t\t\t{
\t\t\t\tmatch = j;
\t\t\t}
\t\t}
\t\tif ( match >= 0 )
\t\t{
\t\t\tkept[match] = 1;
\t\t\tnshape[i] = b2u_terrain_shape[s][match];
\t\t}
\t\telse
\t\t{
\t\t\tb2Polygon box = b2MakeOffsetBox( b[2], b[3], (b2Vec2){ b[0], b[1] }, b2Rot_identity );
\t\t\tnshape[i] = b2CreatePolygonShape( bodyId, &def, &box );
\t\t}
\t\tnbox[i][0] = b[0];
\t\tnbox[i][1] = b[1];
\t\tnbox[i][2] = b[2];
\t\tnbox[i][3] = b[3];
\t}
\tfor ( int j = 0; j < old; ++j )
\t{
\t\tif ( kept[j] == 0 )
\t\t{
\t\t\tb2DestroyShape( b2u_terrain_shape[s][j], false );
\t\t}
\t}
\tfor ( int i = 0; i < n; ++i )
\t{
\t\tb2u_terrain_shape[s][i] = nshape[i];
\t\tb2u_terrain_box[s][i][0] = nbox[i][0];
\t\tb2u_terrain_box[s][i][1] = nbox[i][1];
\t\tb2u_terrain_box[s][i][2] = nbox[i][2];
\t\tb2u_terrain_box[s][i][3] = nbox[i][3];
\t}
\tb2u_terrain_count[s] = n;
}

"""


def _with_terrain( glue, chunks, max_shapes ):
    """
    Terrain chunks (plan["physics2d_terrain"]): collider kind 5, a static body at the collider's
    center without a shape of its own. The game replaces its shapes with

        void b2u_terrain_set( int ci, const float* boxes, int n );

    n boxes of (center x, center y, half width, half height) relative to the body, at most
    plan["terrain2d_max_shapes"] (default 256) of them. A box that is unchanged since the last call
    keeps its shape, so a body resting on it does not see its contact end and begin again. The
    body turns with its collider row's cos / sin (the GameObject's rotation when packed, fixed from then on), and a terrain collider has no Rigidbody2D.
    """
    edits = [
        ( "static void b2u_add_shape( b2BodyId bodyId, int ci, b2Vec2 offset )\n",
          TERRAIN_SHAPES.replace( "{CHUNKS}", str( chunks ) ).replace( "{SHAPES}", str( max_shapes ) )
          + "static void b2u_add_shape( b2BodyId bodyId, int ci, b2Vec2 offset )\n" ),
        ( "\t\tb2u_add_shape( bodyId, ci, b2Vec2_zero );\n\t\tb2u_col_body[ci] = bodyId;\n",
          "\t\tif ( _Collider2D_kind[ci] == %d )\n\t\t\tb2u_terrain_claim( ci );\n\t\telse\n"
          "\t\t\tb2u_add_shape( bodyId, ci, b2Vec2_zero );\n\t\tb2u_col_body[ci] = bodyId;\n" % TERRAIN_KIND ),
        ( "\t\tdef.position = (b2Pos){ x, y };\n\t\tb2BodyId bodyId = b2CreateBody( b2u_world, &def );\n\t\tif ( _Collider2D_kind[ci]",
          "\t\tdef.position = (b2Pos){ x, y };\n"
          "\t\tif ( _Collider2D_kind[ci] == %d )\n\t\t\tdef.rotation = (b2Rot){ _Collider2D_cos[ci], _Collider2D_sin[ci] };\n"
          "\t\tb2BodyId bodyId = b2CreateBody( b2u_world, &def );\n\t\tif ( _Collider2D_kind[ci]" % TERRAIN_KIND ),
        ( "void engine_box2d_step( void )\n{\n", TERRAIN_SET + "void engine_box2d_step( void )\n{\n" ),
    ]
    for old, new in edits:
        if glue.count( old ) != 1:
            raise ValueError( "box2d_unity: terrain anchor not found: %r" % old[:60] )
        glue = glue.replace( old, new )
    return glue


TERRAIN_CHAIN_STORE = """
/* Terrain chains: a chunk's boundary traced from its pixels, ground on the left of each chain.
 * A chain that is the same as in the previous call keeps its shapes. */
#define B2U_TERRAIN_CHAINS {CHAINS}
#define B2U_TERRAIN_POINTS {POINTS}
static int b2u_chain_count[B2U_TERRAIN_CHUNKS];
static int b2u_chain_first[B2U_TERRAIN_CHUNKS][B2U_TERRAIN_CHAINS]; /* into b2u_chain_pt */
static int b2u_chain_len[B2U_TERRAIN_CHUNKS][B2U_TERRAIN_CHAINS];
static int b2u_chain_loop[B2U_TERRAIN_CHUNKS][B2U_TERRAIN_CHAINS];
static float b2u_chain_pt[B2U_TERRAIN_CHUNKS][B2U_TERRAIN_POINTS][2];
static b2ChainId b2u_chain_id[B2U_TERRAIN_CHUNKS][B2U_TERRAIN_CHAINS];

static void b2u_terrain_clear_chains( int s )
{
\tfor ( int k = 0; k < b2u_chain_count[s]; ++k )
\t{
\t\tb2DestroyChain( b2u_chain_id[s][k] );
\t}
\tb2u_chain_count[s] = 0;
}

"""

TERRAIN_CHAINS_SET = """
/* Replace the shapes of terrain chunk ci with chains. pts holds x, y pairs relative to the chunk,
 * chain k is counts[k] points from point starts[k], closed when loops[k] != 0. The ground is on
 * the left of the way, and the chain collides on its right (the air). An open chain's ghost points
 * continue its two end segments. At most B2U_TERRAIN_CHAINS chains and B2U_TERRAIN_POINTS points in
 * all are taken, the rest are cut. */
void b2u_terrain_set_chains( int ci, const float* pts, const int* starts, const int* counts, const int* loops, int n )
{
\tb2u_ensure();
\tif ( ci < 0 || ci >= _Collider2D_count || ci >= B2U_MAX_COL || b2u_terrain_slot[ci] == 0 || b2u_col_has_body[ci] == 0 )
\t{
\t\treturn;
\t}
\tint s = b2u_terrain_slot[ci] - 1;
\t/* no boxes while there are chains */
\tfor ( int j = 0; j < b2u_terrain_count[s]; ++j )
\t{
\t\tb2DestroyShape( b2u_terrain_shape[s][j], false );
\t}
\tb2u_terrain_count[s] = 0;
\tif ( n > B2U_TERRAIN_CHAINS )
\t{
\t\tn = B2U_TERRAIN_CHAINS;
\t}
\tint old = b2u_chain_count[s];

\tstatic unsigned char kept[B2U_TERRAIN_CHAINS];
\tstatic b2ChainId nid[B2U_TERRAIN_CHAINS];
\tstatic int nfirst[B2U_TERRAIN_CHAINS];
\tstatic int nlen[B2U_TERRAIN_CHAINS];
\tstatic int nloop[B2U_TERRAIN_CHAINS];
\tstatic float npt[B2U_TERRAIN_POINTS][2];
\tfor ( int j = 0; j < old; ++j )
\t{
\t\tkept[j] = 0;
\t}

\tb2BodyId bodyId = b2u_col_body[ci];
\tb2SurfaceMaterial material = b2DefaultSurfaceMaterial();
\tmaterial.friction = _Collider2D_friction[ci];
\tmaterial.restitution = _Collider2D_bounciness[ci];
\tmaterial.userMaterialId =
\t\t(uint64_t)( _Collider2D_friction_combine[ci] & 0xff ) | ( (uint64_t)( _Collider2D_bounce_combine[ci] & 0xff ) << 8 );

\tint used = 0, made = 0;
\tfor ( int k = 0; k < n; ++k )
\t{
\t\tint len = counts[k], loop = loops[k] != 0;
\t\tif ( len < ( loop ? 3 : 2 ) || used + len > B2U_TERRAIN_POINTS )
\t\t{
\t\t\tcontinue;
\t\t}
\t\tconst float* p = pts + 2 * starts[k];
\t\tfor ( int i = 0; i < len; ++i )
\t\t{
\t\t\tnpt[used + i][0] = p[2 * i];
\t\t\tnpt[used + i][1] = p[2 * i + 1];
\t\t}
\t\tint match = -1;
\t\tfor ( int j = 0; j < old && match < 0; ++j )
\t\t{
\t\t\tif ( kept[j] || b2u_chain_len[s][j] != len || b2u_chain_loop[s][j] != loop )
\t\t\t{
\t\t\t\tcontinue;
\t\t\t}
\t\t\tint same = 1;
\t\t\tfor ( int i = 0; i < len && same; ++i )
\t\t\t{
\t\t\t\tsame = b2u_chain_pt[s][b2u_chain_first[s][j] + i][0] == p[2 * i] &&
\t\t\t\t\t   b2u_chain_pt[s][b2u_chain_first[s][j] + i][1] == p[2 * i + 1];
\t\t\t}
\t\t\tif ( same )
\t\t\t{
\t\t\t\tmatch = j;
\t\t\t}
\t\t}
\t\tif ( match >= 0 )
\t\t{
\t\t\tkept[match] = 1;
\t\t\tnid[made] = b2u_chain_id[s][match];
\t\t}
\t\telse
\t\t{
\t\t\tstatic b2Vec2 points[B2U_TERRAIN_POINTS];
\t\t\tfor ( int i = 0; i < len; ++i )
\t\t\t{
\t\t\t\tpoints[i] = (b2Vec2){ p[2 * i], p[2 * i + 1] };
\t\t\t}
\t\t\tb2ChainDef def = b2DefaultChainDef();
\t\t\tdef.userData = (void*)(intptr_t)( ci + 1 );
\t\t\tdef.points = points;
\t\t\tdef.pointCount = len;
\t\t\tdef.materials = &material;
\t\t\tdef.materialCount = 1;
\t\t\tdef.isLoop = loop;
\t\t\tif ( loop == 0 )
\t\t\t{
\t\t\t\tdef.ghost1 = b2Sub( points[0], b2Sub( points[1], points[0] ) );
\t\t\t\tdef.ghost2 = b2Add( points[len - 1], b2Sub( points[len - 1], points[len - 2] ) );
\t\t\t}
\t\t\tnid[made] = b2CreateChain( bodyId, &def );
\t\t}
\t\tnfirst[made] = used;
\t\tnlen[made] = len;
\t\tnloop[made] = loop;
\t\tused += len;
\t\tmade += 1;
\t}
\tfor ( int j = 0; j < old; ++j )
\t{
\t\tif ( kept[j] == 0 )
\t\t{
\t\t\tb2DestroyChain( b2u_chain_id[s][j] );
\t\t}
\t}
\tfor ( int k = 0; k < made; ++k )
\t{
\t\tb2u_chain_id[s][k] = nid[k];
\t\tb2u_chain_first[s][k] = nfirst[k];
\t\tb2u_chain_len[s][k] = nlen[k];
\t\tb2u_chain_loop[s][k] = nloop[k];
\t}
\tfor ( int i = 0; i < used; ++i )
\t{
\t\tb2u_chain_pt[s][i][0] = npt[i][0];
\t\tb2u_chain_pt[s][i][1] = npt[i][1];
\t}
\tb2u_chain_count[s] = made;
}

"""


def _with_terrain_chains( glue, max_chains, max_points ):
    """
    Terrain chains (plan["terrain2d_chains"], on top of _with_terrain): a chunk's ground boundary as
    Box2D chains, which a solid region makes one of however large it is, and which are smooth for
    things rolling over them:

        void b2u_terrain_set_chains( int ci, const float* pts, const int* starts,
                                     const int* counts, const int* loops, int n );

    A chunk holds boxes or chains, whichever was set last. A chain unchanged since the last call keeps
    its shapes (a body resting on it keeps its contact); the others are destroyed and made. The edges
    must be longer than Box2D's linear slop (0.005 m), so pixels per unit must stay under 200.
    """
    edits = [
        ( "/* Replace the shapes of terrain chunk ci with n boxes",
          TERRAIN_CHAIN_STORE.replace( "{CHAINS}", str( max_chains ) ).replace( "{POINTS}", str( max_points ) )
          + "/* Replace the shapes of terrain chunk ci with n boxes" ),
        ( "\tint s = b2u_terrain_slot[ci] - 1;\n\tint old = b2u_terrain_count[s];\n",
          "\tint s = b2u_terrain_slot[ci] - 1;\n\tb2u_terrain_clear_chains( s );\n\tint old = b2u_terrain_count[s];\n" ),
        ( "void engine_box2d_step( void )\n{\n", TERRAIN_CHAINS_SET + "void engine_box2d_step( void )\n{\n" ),
    ]
    for old, new in edits:
        if glue.count( old ) != 1:
            raise ValueError( "box2d_unity: terrain chain anchor not found: %r" % old[:60] )
        glue = glue.replace( old, new )
    return glue


def _write_if_different( path, text ):
    try:
        with open( path, encoding="utf-8" ) as f:
            if f.read() == text:
                return
    except OSError:
        pass
    with open( path, "w", encoding="utf-8" ) as f:
        f.write( text )


COLLIDER_EXTERNS = """extern const int _Collider2D_count;
extern const int _Collider2D_kind[];
extern const int _Collider2D_is_trigger[];
extern const int _Collider2D_rb2d[];
extern const int _Collider2D_owner_class[];
extern const int _Collider2D_owner_inst[];
extern const float _Collider2D_ox[];
extern const float _Collider2D_oy[];
extern const float _Collider2D_hw[];
extern const float _Collider2D_hh[];
extern const float _Collider2D_edge_r[];
extern const float _Collider2D_cos[];
extern const float _Collider2D_sin[];
extern const float _Collider2D_friction[];
extern const float _Collider2D_bounciness[];
extern const int _Collider2D_friction_combine[];
extern const int _Collider2D_bounce_combine[];
extern const int _Collider2D_layer[];
"""

CONTACT_EXPORTS = """void engine_col2d_manifold( int a, int b, float nx, float ny, int n, float p0x, float p0y,
							float p1x, float p1y );
"""

CONTACT_STATE = """
/* Each collider's shape, to read a touching pair's manifold after the step */
static b2ShapeId b2u_col_shape[B2U_MAX_COL];
static int b2u_col_has_shape[B2U_MAX_COL];

/* Report the manifold of pair (a, b): normal from a to b, world points */
static void b2u_report_manifold( int a, int b )
{
	b2ContactData data[32];
	if ( a < 0 || b < 0 || b2u_col_has_shape[a] == 0 || b2u_col_has_shape[b] == 0 )
		return;
	int n = b2Shape_GetContactData( b2u_col_shape[a], data, 32 );
	for ( int k = 0; k < n; ++k )
	{
		int flip;
		if ( B2_ID_EQUALS( data[k].shapeIdA, b2u_col_shape[a] ) && B2_ID_EQUALS( data[k].shapeIdB, b2u_col_shape[b] ) )
			flip = 0;
		else if ( B2_ID_EQUALS( data[k].shapeIdA, b2u_col_shape[b] ) && B2_ID_EQUALS( data[k].shapeIdB, b2u_col_shape[a] ) )
			flip = 1;
		else
			continue;
		b2Manifold* m = &data[k].manifold;
		b2Pos center = b2Body_GetWorldCenter( b2Shape_GetBody( data[k].shapeIdA ) );
		float px[2] = { 0.0f, 0.0f }, py[2] = { 0.0f, 0.0f };
		int count = m->pointCount < 2 ? m->pointCount : 2;
		for ( int q = 0; q < count; ++q )
		{
			b2Pos w = b2OffsetPos( center, m->points[q].anchorA );
			px[q] = (float)w.x;
			py[q] = (float)w.y;
		}
		float nx = flip ? -m->normal.x : m->normal.x;
		float ny = flip ? -m->normal.y : m->normal.y;
		engine_col2d_manifold( a, b, nx, ny, count, px[0], py[0], px[1], py[1] );
		return;
	}
}
"""

LIVE_EXPORTS = """int engine_rb2d_live( int rb );
int engine_col2d_live( int ci );
"""

LIVE_STATE = """
/* Which bodies are in the simulation (engine_rb2d_live / engine_col2d_live) */
static int b2u_rb_on[B2U_MAX_RB];
static int b2u_col_on[B2U_MAX_COL];
/* A Rigidbody2D's collider on an inactive GameObject: filtered out of contacts
 * and queries (destroying the shape would recompute the authored mass) */
static int b2u_col_gone[B2U_MAX_COL];
static b2Filter b2u_col_filter[B2U_MAX_COL];

/* A disabled body's contacts are gone: forget its touching pairs */
static void b2u_drop_pairs( int rb, int ci )
{
	int k = 0;
	for ( int i = 0; i < b2u_pair_n; ++i )
	{
		int a = b2u_pair_a[i], b = b2u_pair_b[i];
		int hit = a == ci || b == ci;
		if ( rb >= 0 )
		{
			hit = hit || ( a < _Collider2D_count && _Collider2D_rb2d[a] == rb ) ||
				  ( b < _Collider2D_count && _Collider2D_rb2d[b] == rb );
		}
		if ( hit )
			continue;
		b2u_pair_a[k] = a;
		b2u_pair_b[k] = b;
		k += 1;
	}
	b2u_pair_n = k;
}
"""

LIVE_SYNC = """\t/* Bodies of GameObjects that left or rejoined the simulation */
\tfor ( int rb = 0; rb < b2u_rb_created; ++rb )
\t{
\t\tint on = engine_rb2d_live( rb ) != 0;
\t\tif ( on == b2u_rb_on[rb] )
\t\t\tcontinue;
\t\tb2u_rb_on[rb] = on;
\t\tif ( on )
\t\t{
\t\t\tb2Body_Enable( b2u_rb_body[rb] );
\t\t}
\t\telse
\t\t{
\t\t\tb2Body_Disable( b2u_rb_body[rb] );
\t\t\tb2u_drop_pairs( rb, -1 );
\t\t}
\t}
\tfor ( int ci = 0; ci < _Collider2D_count && ci < B2U_MAX_COL; ++ci )
\t{
\t\tif ( b2u_col_has_body[ci] == 0 )
\t\t\tcontinue;
\t\tint on = engine_col2d_live( ci ) != 0;
\t\tif ( on == b2u_col_on[ci] )
\t\t\tcontinue;
\t\tb2u_col_on[ci] = on;
\t\tif ( on )
\t\t{
\t\t\tb2Body_Enable( b2u_col_body[ci] );
\t\t}
\t\telse
\t\t{
\t\t\tb2Body_Disable( b2u_col_body[ci] );
\t\t\tb2u_drop_pairs( -1, ci );
\t\t}
\t}
\tfor ( int rb = 0; rb < b2u_rb_created; ++rb )
\t{
\t\t/* ponytail: the first 32 shapes of a body; raise the cap for bigger compounds */
\t\tb2ShapeId shapes[32];
\t\tint n = b2Body_GetShapes( b2u_rb_body[rb], shapes, 32 );
\t\tfor ( int k = 0; k < n; ++k )
\t\t{
\t\t\tint ci = (int)(intptr_t)b2Shape_GetUserData( shapes[k] ) - 1;
\t\t\tif ( ci < 0 || ci >= _Collider2D_count || ci >= B2U_MAX_COL )
\t\t\t\tcontinue;
\t\t\tint gone = engine_col2d_live( ci ) == 0;
\t\t\tif ( gone == b2u_col_gone[ci] )
\t\t\t\tcontinue;
\t\t\tb2u_col_gone[ci] = gone;
\t\t\tif ( gone )
\t\t\t{
\t\t\t\tb2Filter f = b2u_col_filter[ci] = b2Shape_GetFilter( shapes[k] );
\t\t\t\tf.categoryBits = 0;
\t\t\t\tf.maskBits = 0;
\t\t\t\tb2Shape_SetFilter( shapes[k], f );
\t\t\t\tb2u_drop_pairs( -1, ci );
\t\t\t}
\t\t\telse
\t\t\t{
\t\t\t\tb2Shape_SetFilter( shapes[k], b2u_col_filter[ci] );
\t\t\t}
\t\t}
\t}
"""

GLUE_TEMPLATE = r"""/* Generated by box2d_unity.py for {PACKER}. Do not edit. */

#include "box2d/box2d.h"

#include <stdint.h>
{EXTRA_INCLUDES}
#ifndef B2_PACK_INJECTED
#define B2_PACK_INJECTED 0
#endif

/* unity_pack tables (data.c) */
extern float Time_fixedDeltaTime;
extern float Physics2D_gravity_x;
extern float Physics2D_gravity_y;
extern int _Rigidbody2D_count;
extern float _Rigidbody2D_vel_x[];
extern float _Rigidbody2D_vel_y[];
extern float _Rigidbody2D_gravity_scale[];
extern float _Rigidbody2D_linear_damping[];
extern float _Rigidbody2D_mass[];
extern int _Rigidbody2D_body_type[];
extern int _Rigidbody2D_owner_class[];
extern int _Rigidbody2D_owner_inst[];
{COLLIDER_DECLS}
/* unity_pack exports (engine.c) */
void engine_rb2d_get_pos( int rb, float* x, float* y );
void engine_rb2d_set_pos( int rb, float x, float y );
void engine_col2d_center( int ci, float* x, float* y );
void engine_col2d_contact( int a, int b );

void engine_box2d_step( void );
void b2u_on_begin( int colliderA, int colliderB );
void b2u_on_end( int colliderA, int colliderB );

enum
{{
	B2U_MAX_RB = {N_RB},
	B2U_MAX_COL = {N_COL},
	B2U_MAX_PAIRS = {MAX_PAIRS},
	B2U_SUB_STEPS = {SUB_STEPS},
}};

static int b2u_ready;
static int b2u_rb_created;
static b2WorldId b2u_world;
static b2BodyId b2u_rb_body[B2U_MAX_RB];
static float b2u_last_x[B2U_MAX_RB];
static float b2u_last_y[B2U_MAX_RB];
/* Rigidbody2D.mass / bodyType as last pushed: a script may change them */
static float b2u_last_mass[B2U_MAX_RB];
static int b2u_last_type[B2U_MAX_RB];

/* Static bodies of the colliders without a Rigidbody2D */
static b2BodyId b2u_col_body[B2U_MAX_COL];
static int b2u_col_has_body[B2U_MAX_COL];

/* Touching collider pairs, lo < hi, maintained from contact begin and end */
static int b2u_pair_a[B2U_MAX_PAIRS];
static int b2u_pair_b[B2U_MAX_PAIRS];
static int b2u_pair_n;

void b2u_on_begin( int colliderA, int colliderB )
{{
	int lo = colliderA < colliderB ? colliderA : colliderB;
	int hi = colliderA < colliderB ? colliderB : colliderA;
	if ( lo < 0 || lo == hi )
	{{
		return;
	}}
	for ( int i = 0; i < b2u_pair_n; ++i )
	{{
		if ( b2u_pair_a[i] == lo && b2u_pair_b[i] == hi )
		{{
			return;
		}}
	}}
	if ( b2u_pair_n < B2U_MAX_PAIRS )
	{{
		b2u_pair_a[b2u_pair_n] = lo;
		b2u_pair_b[b2u_pair_n] = hi;
		b2u_pair_n += 1;
	}}
}}

void b2u_on_end( int colliderA, int colliderB )
{{
	int lo = colliderA < colliderB ? colliderA : colliderB;
	int hi = colliderA < colliderB ? colliderB : colliderA;
	for ( int i = 0; i < b2u_pair_n; ++i )
	{{
		if ( b2u_pair_a[i] == lo && b2u_pair_b[i] == hi )
		{{
			/* Keep order stable, messages are sent in pair order */
			for ( int k = i + 1; k < b2u_pair_n; ++k )
			{{
				b2u_pair_a[k - 1] = b2u_pair_a[k];
				b2u_pair_b[k - 1] = b2u_pair_b[k];
			}}
			b2u_pair_n -= 1;
			return;
		}}
	}}
}}

{MATERIALS}static b2BodyType b2u_body_type( int unityType )
{{
	/* Rigidbody2D.bodyType: Dynamic=0 Kinematic=1 Static=2 */
	if ( unityType == 1 )
		return b2_kinematicBody;
	if ( unityType == 2 )
		return b2_staticBody;
	return b2_dynamicBody;
}}

static void b2u_add_shape( b2BodyId bodyId, int ci, b2Vec2 offset )
{{
	b2ShapeDef def = b2DefaultShapeDef();
	def.userData = (void*)(intptr_t)( ci + 1 );
	def.material.friction = _Collider2D_friction[ci];
	def.material.restitution = _Collider2D_bounciness[ci];
	def.material.userMaterialId =
		(uint64_t)( _Collider2D_friction_combine[ci] & 0xff ) | ( (uint64_t)( _Collider2D_bounce_combine[ci] & 0xff ) << 8 );
	def.isSensor = _Collider2D_is_trigger[ci] != 0;
	def.enableContactEvents = _Collider2D_is_trigger[ci] == 0;
{SHAPE_EXTRA}
	b2Rot rotation = {{ _Collider2D_cos[ci], _Collider2D_sin[ci] }};
	/* CapsuleCollider2D: kind 2 vertical, 3 horizontal. One no longer than it
	 * is wide is a circle, as in Unity (Box2D refuses a zero-length capsule). */
	int cap = _Collider2D_kind[ci] >= 2, vert = _Collider2D_kind[ci] == 2;
	float r = !cap || vert ? _Collider2D_hw[ci] : _Collider2D_hh[ci];
	float h = cap ? ( vert ? _Collider2D_hh[ci] : _Collider2D_hw[ci] ) - r : 0.0f;
	if ( _Collider2D_kind[ci] == 1 || ( cap && h <= 0.005f ) )
	{{
		b2Circle circle = {{ offset, r }};
		b2CreateCircleShape( bodyId, &def, &circle );
	}}
	else if ( cap )
	{{
		b2Vec2 d = b2RotateVector( rotation, vert ? (b2Vec2){{ 0.0f, h }} : (b2Vec2){{ h, 0.0f }} );
		b2Capsule capsule = {{ b2Sub( offset, d ), b2Add( offset, d ), r }};
		b2CreateCapsuleShape( bodyId, &def, &capsule );
	}}
	else
	{{
		/* BoxCollider2D.edgeRadius rounds the box outward */
		float er = _Collider2D_edge_r[ci] > 0.0f ? _Collider2D_edge_r[ci] : {POLY_RADIUS};
		b2Polygon box = b2MakeOffsetRoundedBox( _Collider2D_hw[ci], _Collider2D_hh[ci], offset, rotation, er );
		b2CreatePolygonShape( bodyId, &def, &box );
	}}
}}

/* Body for Rigidbody2D rb, with the colliders the scene attached to it */
static void b2u_create_body( int rb )
{{
	float x, y;
	engine_rb2d_get_pos( rb, &x, &y );
	b2BodyDef def = b2DefaultBodyDef();
	def.type = b2u_body_type( _Rigidbody2D_body_type[rb] );
	def.position = (b2Pos){{ x, y }};
	def.linearVelocity = (b2Vec2){{ _Rigidbody2D_vel_x[rb], _Rigidbody2D_vel_y[rb] }};
	def.gravityScale = _Rigidbody2D_gravity_scale[rb];
	def.linearDamping = {LINEAR_DAMPING};
	def.motionLocks.angularZ = true;
	b2BodyId bodyId = b2CreateBody( b2u_world, &def );
	b2u_rb_body[rb] = bodyId;
	b2u_last_x[rb] = x;
	b2u_last_y[rb] = y;

	for ( int ci = 0; ci < _Collider2D_count && ci < B2U_MAX_COL; ++ci )
	{{
		if ( _Collider2D_rb2d[ci] != rb )
			continue;
		/* Offset relative to the body origin, rotated like the collider */
		float c = _Collider2D_cos[ci], s = _Collider2D_sin[ci];
		float ox = _Collider2D_ox[ci], oy = _Collider2D_oy[ci];
		b2Vec2 off = {{ c * ox - s * oy, s * ox + c * oy }};
		/* A child GameObject's collider without its own Rigidbody2D sits on the
		 * ancestor's body, away from the body origin (Unity) */
		if ( _Collider2D_owner_class[ci] != _Rigidbody2D_owner_class[rb] ||
			 _Collider2D_owner_inst[ci] != _Rigidbody2D_owner_inst[rb] )
		{{
			float cx, cy;
			engine_col2d_center( ci, &cx, &cy );
			off = b2InvRotateVector( def.rotation, (b2Vec2){{ cx - x, cy - y }} );
		}}
		b2u_add_shape( bodyId, ci, off );
	}}

	/* Rigidbody2D.mass: scale the shape-derived mass data to the authored mass */
	if ( _Rigidbody2D_body_type[rb] == 0 && _Rigidbody2D_mass[rb] > 0.0f )
	{{
		b2MassData md = b2Body_GetMassData( bodyId );
		float mass = _Rigidbody2D_mass[rb];
		if ( md.mass > 0.0f )
		{{
			md.rotationalInertia *= mass / md.mass;
		}}
		md.mass = mass;
		b2Body_SetMassData( bodyId, md );
	}}
	b2u_last_mass[rb] = _Rigidbody2D_mass[rb];
	b2u_last_type[rb] = _Rigidbody2D_body_type[rb];
}}

static void b2u_create( void )
{{
{WORLD_PRELUDE}	b2WorldDef worldDef = b2DefaultWorldDef();
	worldDef.gravity = (b2Vec2){{ Physics2D_gravity_x, Physics2D_gravity_y }};
	worldDef.frictionCallback = b2u_friction;
	worldDef.restitutionCallback = b2u_restitution;
	b2u_world = b2CreateWorld( &worldDef );

	/* Colliders without a Rigidbody2D are static bodies at the collider center */
	for ( int ci = 0; ci < _Collider2D_count && ci < B2U_MAX_COL; ++ci )
	{{
		int rb = _Collider2D_rb2d[ci];
		if ( rb >= 0 && rb < B2U_MAX_RB )
			continue;
		float x, y;
		engine_col2d_center( ci, &x, &y );
		b2BodyDef def = b2DefaultBodyDef();
		def.position = (b2Pos){{ x, y }};
		b2BodyId bodyId = b2CreateBody( b2u_world, &def );
		b2u_add_shape( bodyId, ci, b2Vec2_zero );
		b2u_col_body[ci] = bodyId;
		b2u_col_has_body[ci] = 1;
	}}

	b2u_rb_created = 0;
	b2u_pair_n = 0;
	b2u_ready = 1;
}}

/* The world, and every body so far: on the first step, or a query before it
 * (a script's Start), and AddComponent<Rigidbody2D> bodies when they appear */
static void b2u_ensure( void )
{{
	if ( b2u_ready == 0 )
	{{
		b2u_create();
	}}
	while ( b2u_rb_created < _Rigidbody2D_count && b2u_rb_created < B2U_MAX_RB )
	{{
		b2u_create_body( b2u_rb_created );
		b2u_rb_created += 1;
	}}
}}

void engine_box2d_step( void )
{{
	b2u_ensure();

	/* A static collider whose Transform moved (a parent, a script) is teleported, as in Unity */
	for ( int ci = 0; ci < _Collider2D_count && ci < B2U_MAX_COL; ++ci )
	{{
		if ( b2u_col_has_body[ci] == 0 )
			continue;
		float x, y;
		engine_col2d_center( ci, &x, &y );
		b2BodyId bodyId = b2u_col_body[ci];
		b2Pos p = b2Body_GetPosition( bodyId );
		if ( p.x != x || p.y != y )
			b2Body_SetTransform( bodyId, (b2Pos){{ x, y }}, b2Body_GetRotation( bodyId ) );
	}}

	/* Push what scripts may have changed since the last step */
	b2World_SetGravity( b2u_world, (b2Vec2){{ Physics2D_gravity_x, Physics2D_gravity_y }} );
	for ( int rb = 0; rb < b2u_rb_created; ++rb )
	{{
		b2BodyId bodyId = b2u_rb_body[rb];
		float x, y;
		engine_rb2d_get_pos( rb, &x, &y );
		if ( x != b2u_last_x[rb] || y != b2u_last_y[rb] )
		{{
			/* transform.position written by a script */
			b2Body_SetTransform( bodyId, (b2Pos){{ x, y }}, b2Rot_identity );
		}}
		if ( _Rigidbody2D_body_type[rb] != b2u_last_type[rb] )
		{{
			/* Rigidbody2D.bodyType / isKinematic written by a script */
			b2Body_SetType( bodyId, b2u_body_type( _Rigidbody2D_body_type[rb] ) );
			b2u_last_type[rb] = _Rigidbody2D_body_type[rb];
			b2u_last_mass[rb] = -1.0f;
		}}
		if ( _Rigidbody2D_body_type[rb] == 0 && _Rigidbody2D_mass[rb] > 0.0f &&
			 _Rigidbody2D_mass[rb] != b2u_last_mass[rb] )
		{{
			/* Rigidbody2D.mass written by a script: the mass data scaled to it */
			b2MassData md = b2Body_GetMassData( bodyId );
			float mass = _Rigidbody2D_mass[rb];
			if ( md.mass > 0.0f )
			{{
				md.rotationalInertia *= mass / md.mass;
			}}
			md.mass = mass;
			b2Body_SetMassData( bodyId, md );
			b2u_last_mass[rb] = mass;
		}}
		if ( _Rigidbody2D_body_type[rb] != 2 )
		{{
			b2Body_SetLinearVelocity( bodyId, (b2Vec2){{ _Rigidbody2D_vel_x[rb], _Rigidbody2D_vel_y[rb] }} );
		}}
		if ( _Rigidbody2D_body_type[rb] == 0 )
		{{
			b2Body_SetGravityScale( bodyId, _Rigidbody2D_gravity_scale[rb] );
			b2Body_SetLinearDamping( bodyId, {LINEAR_DAMPING} );
		}}
	}}

	float dt = Time_fixedDeltaTime > 1e-8f ? Time_fixedDeltaTime : {DEFAULT_DT};
	b2World_Step( b2u_world, dt, B2U_SUB_STEPS );

#if B2_PACK_INJECTED == 0
	/* Standard API: touching pairs from the contact event arrays */
	b2ContactEvents events = b2World_GetContactEvents( b2u_world );
	for ( int i = 0; i < events.beginCount; ++i )
	{{
		b2ContactBeginTouchEvent* e = events.beginEvents + i;
		b2u_on_begin( (int)(intptr_t)b2Shape_GetUserData( e->shapeIdA ) - 1,
					  (int)(intptr_t)b2Shape_GetUserData( e->shapeIdB ) - 1 );
	}}
	for ( int i = 0; i < events.endCount; ++i )
	{{
		b2ContactEndTouchEvent* e = events.endEvents + i;
		if ( b2Shape_IsValid( e->shapeIdA ) && b2Shape_IsValid( e->shapeIdB ) )
		{{
			b2u_on_end( (int)(intptr_t)b2Shape_GetUserData( e->shapeIdA ) - 1,
						(int)(intptr_t)b2Shape_GetUserData( e->shapeIdB ) - 1 );
		}}
	}}
{SENSOR_EVENTS}#endif

	/* Pull positions and velocities into the packed tables */
	for ( int rb = 0; rb < b2u_rb_created; ++rb )
	{{
		b2BodyId bodyId = b2u_rb_body[rb];
		b2Pos p = b2Body_GetPosition( bodyId );
		b2Vec2 v = b2Body_GetLinearVelocity( bodyId );
		float x = (float)p.x, y = (float)p.y;
		engine_rb2d_set_pos( rb, x, y );
		b2u_last_x[rb] = x;
		b2u_last_y[rb] = y;
		_Rigidbody2D_vel_x[rb] = v.x;
		_Rigidbody2D_vel_y[rb] = v.y;
	}}

	/* unity_pack sends Enter / Stay / Exit by comparing with the previous step */
	for ( int i = 0; i < b2u_pair_n; ++i )
	{{
		engine_col2d_contact( b2u_pair_a[i], b2u_pair_b[i] );
	}}
}}
"""


UNITY_MATERIALS = r"""/* PhysicsMaterialCombine: Average=0 Multiply=1 Minimum=2 Maximum=3, the higher mode wins */
static float b2u_combine( float a, float b, int ca, int cb )
{{
	int mode = ca > cb ? ca : cb;
	if ( mode > 3 )
		mode = 0;
	if ( mode == 1 )
		return a * b;
	if ( mode == 2 )
		return a < b ? a : b;
	if ( mode == 3 )
		return a > b ? a : b;
	return 0.5f * ( a + b );
}}

/* userMaterialId carries the combine modes: friction in bits 0-7, bounce in bits 8-15 */
static float b2u_friction( float a, uint64_t ma, float b, uint64_t mb )
{{
	return b2u_combine( a, b, (int)( ma & 0xff ), (int)( mb & 0xff ) );
}}

static float b2u_restitution( float a, uint64_t ma, float b, uint64_t mb )
{{
	return b2u_combine( a, b, (int)( ( ma >> 8 ) & 0xff ), (int)( ( mb >> 8 ) & 0xff ) );
}}

"""

GODOT_MATERIALS = r"""/* Godot's PhysicsMaterial: friction |min(a, b)| and bounce clamp(a + b, 0, 1), where a rough
   material's friction and an absorbent material's bounce count negative (so rough wins the min
   and absorbent subtracts). userMaterialId: rough in bits 0-7, absorbent in bits 8-15 */
static float b2u_friction( float a, uint64_t ma, float b, uint64_t mb )
{{
	float fa = ( ma & 0xff ) != 0 ? -a : a;
	float fb = ( mb & 0xff ) != 0 ? -b : b;
	float f = fa < fb ? fa : fb;
	return f < 0.0f ? -f : f;
}}

static float b2u_restitution( float a, uint64_t ma, float b, uint64_t mb )
{{
	float ba = ( ( ma >> 8 ) & 0xff ) != 0 ? -a : a;
	float bb = ( ( mb >> 8 ) & 0xff ) != 0 ? -b : b;
	float r = ba + bb;
	if ( r < 0.0f )
		return 0.0f;
	if ( r > 1.0f )
		return 1.0f;
	return r;
}}

/* Godot damps once a step, v *= max(0, 1 - dt * d); Box2D once a substep, v *= 1 / (1 + h * c).
   The c whose B2U_SUB_STEPS substeps compound to Godot's factor for the step: */
static float b2g_linear_damping( float d )
{{
	float dt = Time_fixedDeltaTime > 1e-8f ? Time_fixedDeltaTime : ( 1.0f / 60.0f );
	float h = dt / (float)B2U_SUB_STEPS;
	float f = 1.0f - dt * d;
	if ( d <= 0.0f )
		return 0.0f;
	if ( f <= 1e-6f )
		return 1e6f / h; /* stopped within the step */
	return ( powf( f, -1.0f / (float)B2U_SUB_STEPS ) - 1.0f ) / h;
}}

"""


GODOT_SENSOR_EVENTS = """\t/* Godot: Area2D overlaps, reported as touching pairs like contacts */
\tb2SensorEvents sensors = b2World_GetSensorEvents( b2u_world );
\tfor ( int i = 0; i < sensors.beginCount; ++i )
\t{
\t\tb2SensorBeginTouchEvent* e = sensors.beginEvents + i;
\t\tb2u_on_begin( (int)(intptr_t)b2Shape_GetUserData( e->sensorShapeId ) - 1,
\t\t\t\t\t  (int)(intptr_t)b2Shape_GetUserData( e->visitorShapeId ) - 1 );
\t}
\tfor ( int i = 0; i < sensors.endCount; ++i )
\t{
\t\tb2SensorEndTouchEvent* e = sensors.endEvents + i;
\t\tif ( b2Shape_IsValid( e->sensorShapeId ) && b2Shape_IsValid( e->visitorShapeId ) )
\t\t{
\t\t\tb2u_on_end( (int)(intptr_t)b2Shape_GetUserData( e->sensorShapeId ) - 1,
\t\t\t\t\t\t(int)(intptr_t)b2Shape_GetUserData( e->visitorShapeId ) - 1 );
\t\t}
\t}
"""


def _with_godot_layers( glue ):
    """Godot's collision layers (godot_pack's _Collider2D_layer_bits / _mask_bits, 32 bits
    each), as Godot's physics decides a body pair (godot_body_pair_2d.cpp): a dynamic body is
    pushed by what its mask has the layer of. A Box2D contact pushes both of a pair's bodies, so
    the pair collides when either dynamic side's mask has the other's layer -- exact whenever one
    side is static or kinematic (godot_pack warns about the rest). A sensor (an Area2D) sees
    every shape: which overlaps an area reports is godot_pack's dispatch's (its mask)."""
    glue = glue.replace(
        "void engine_box2d_step( void );\n",
        "void engine_box2d_step( void );\n"
        "extern const unsigned _Collider2D_layer_bits[];\n"
        "extern const unsigned _Collider2D_mask_bits[];\n", 1 )
    filt = (
        "/* Godot's collision layers: a pair collides when a dynamic side's mask has the other's\n"
        " * layer (the side Godot pushes); a sensor sees all, its reports filtered by godot_pack */\n"
        "static bool b2g_layer_filter( b2ShapeId a, b2ShapeId b, void* context )\n"
        "{\n"
        "\t(void)context;\n"
        "\tif ( b2Shape_IsSensor( a ) || b2Shape_IsSensor( b ) )\n"
        "\t\treturn true;\n"
        "\tint ca = (int)(intptr_t)b2Shape_GetUserData( a ) - 1;\n"
        "\tint cb = (int)(intptr_t)b2Shape_GetUserData( b ) - 1;\n"
        "\tif ( ca < 0 || cb < 0 )\n"
        "\t\treturn true;\n"
        "\tint dynA = b2Body_GetType( b2Shape_GetBody( a ) ) == b2_dynamicBody;\n"
        "\tint dynB = b2Body_GetType( b2Shape_GetBody( b ) ) == b2_dynamicBody;\n"
        "\treturn ( dynA && ( _Collider2D_mask_bits[ca] & _Collider2D_layer_bits[cb] ) != 0 ) ||\n"
        "\t\t   ( dynB && ( _Collider2D_mask_bits[cb] & _Collider2D_layer_bits[ca] ) != 0 );\n"
        "}\n\n" )
    anchor = "static void b2u_add_shape( b2BodyId bodyId, int ci, b2Vec2 offset )\n"
    if anchor not in glue:
        raise ValueError( "box2d_unity: no shape creation to add Godot's layers to" )
    glue = glue.replace( anchor, filt + anchor, 1 )
    glue = glue.replace(
        "\tdef.isSensor = _Collider2D_is_trigger[ci] != 0;\n",
        "\tdef.isSensor = _Collider2D_is_trigger[ci] != 0;\n"
        "\tdef.enableCustomFiltering = true; /* Godot's layers: b2g_layer_filter */\n", 1 )
    glue = glue.replace(
        "\tb2u_world = b2CreateWorld( &worldDef );\n",
        "\tb2u_world = b2CreateWorld( &worldDef );\n"
        "\tb2World_SetCustomFilterCallback( b2u_world, b2g_layer_filter, NULL );\n", 1 )
    return glue


def _mode_parts( mode, length_units_per_meter ):
    """The pieces of GLUE_TEMPLATE that differ by engine. unity's reproduce the glue as it was."""
    if mode == "unity":
        return {
            "PACKER": "unity_pack",
            "EXTRA_INCLUDES": "",
            "MATERIALS": UNITY_MATERIALS.replace( "{{", "{" ).replace( "}}", "}" ),
            "LINEAR_DAMPING": "_Rigidbody2D_linear_damping[rb]",
            "WORLD_PRELUDE": "",
            "DEFAULT_DT": "0.02f",
            "SHAPE_EXTRA": "",
            "SENSOR_EVENTS": "",
            # Unity's polygon skin (Physics2D.defaultContactOffset): boxes rest
            # that far apart, so bounds-edge linecasts miss the floor.
            # ponytail: the default 0.01, not Physics2DSettings' m_DefaultContactOffset
            "POLY_RADIUS": "0.01f",
        }
    return {
        "PACKER": "godot_pack",
        "EXTRA_INCLUDES": "#include <math.h>\n",
        "MATERIALS": GODOT_MATERIALS.replace( "{{", "{" ).replace( "}}", "}" ),
        "LINEAR_DAMPING": "b2g_linear_damping( _Rigidbody2D_linear_damping[rb] )",
        "WORLD_PRELUDE": (
            "\t/* Godot's units are pixels: Box2D's tolerances and default speeds scale to them.\n"
            "\t   Set before any b2Default*Def, which read it */\n"
            "\tb2SetLengthUnitsPerMeter( %sf );\n" % repr( float( length_units_per_meter ) ) ),
        "DEFAULT_DT": "( 1.0f / 60.0f )",
        "SHAPE_EXTRA": (
            "\t/* Godot: areas see every body and area; both shapes take sensor events */\n"
            "\tdef.enableSensorEvents = true;\n" ),
        "SENSOR_EVENTS": GODOT_SENSOR_EVENTS,
        "POLY_RADIUS": "0.0f",
    }
