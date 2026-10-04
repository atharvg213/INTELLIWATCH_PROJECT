/**
 * Scene Graph and Situational Awareness Service
 * Connects UI directly to CuriousPARC backend scene graph endpoints.
 * Strictly presents factual entity detections, directed spatial/compliance relationships,
 * and empirical risk reasoning without fabrication.
 */

import { apiClient } from "../api/client.js";
import { ENDPOINTS } from "../api/endpoints.js";

export const sceneService = {
  /**
   * Fetches the current or job-specific structured Scene Graph.
   * @param {string|null} jobId - Optional analysis job ID to retrieve its specific frame scene.
   * @returns {Promise<{ok: boolean, data?: object, error?: string}>}
   */
  async getSceneGraph(jobId = null) {
    const url = jobId ? ENDPOINTS.SCENE_JOB(jobId) : ENDPOINTS.SCENE_CURRENT;
    const res = await apiClient.get(url);
    if (!res.ok) {
      return {
        ok: false,
        error: res.error || "Unable to load scene graph from backend",
        data: null,
      };
    }

    const raw = res.data || {};

    // Normalize entities: prefer raw.entities if provided, fallback to converting raw.nodes
    let entities = Array.isArray(raw.entities) ? raw.entities : [];
    if (entities.length === 0 && Array.isArray(raw.nodes)) {
      entities = raw.nodes.map((n) => {
        const ntype = n.node_type || "OBJECT";
        let label = n.class_name || "Entity";
        if (ntype === "PERSON") label = n.track_id !== null && n.track_id !== undefined ? (n.attributes?.display_label || `Person #${n.attributes?.display_id || n.track_id}`) : `Person (${n.node_id})`;
        else if (ntype === "VEHICLE") label = n.track_id !== null && n.track_id !== undefined ? `${n.class_name} #${n.track_id}` : n.class_name;
        else if (ntype === "ZONE") label = n.attributes?.name || `Zone ${n.zone_id || n.node_id}`;

        return {
          id: n.node_id,
          type: ntype,
          label,
          class_name: n.class_name,
          track_id: n.track_id,
          confidence: n.confidence !== undefined ? n.confidence : null,
          position: {
            bbox: n.bbox || null,
            centroid: n.centroid || null,
            contact_point: n.contact_point || null,
          },
          state: n.attributes || {},
          risk: {
            level: "INFO",
            score: 0.0,
            factors: [],
            reasons: [],
            active_events: [],
          },
        };
      });
    }

    // Normalize relationships: ensure canonical fields and aliases are populated
    const relationships = (raw.relationships || []).map((r) => {
      const rtype = r.type || r.relation_type || r.relationship || "ASSOCIATED_WITH";
      const rstate = r.state || r.lifecycle || "ACTIVE";
      const srcId = r.source_id || r.source_node_id || r.source;
      const tgtId = r.target_id || r.target_node_id || r.target;
      const relId = r.id || r.relation_id || `rel_${srcId}_${rtype.toLowerCase()}_${tgtId}`;
      const isActive = r.active !== undefined ? Boolean(r.active) : ["ACTIVE", "CREATED"].includes(String(rstate).toUpperCase());

      let category = r.category;
      if (!category) {
        const u = String(rtype).toUpperCase();
        if (["IS_WEARING", "WEARING", "MISSING"].includes(u)) category = "EQUIPMENT";
        else if (["APPROACHING", "MOVING_AWAY", "CLOSER_THAN"].includes(u)) category = "TEMPORAL";
        else if (["INSIDE", "INSIDE_ZONE"].includes(u)) category = "SAFETY";
        else if (["NEAR", "FAR", "ADJACENT"].includes(u)) category = "SPATIAL";
        else category = "SAFETY";
      }

      return {
        id: relId,
        relation_id: relId,
        source_id: srcId,
        target_id: tgtId,
        source: srcId,
        target: tgtId,
        type: rtype,
        relationship: rtype,
        state: rstate,
        active: isActive,
        category,
        confidence: r.confidence !== undefined ? r.confidence : 1.0,
        duration_seconds: r.duration_seconds || 0.0,
        first_seen_timestamp: r.first_seen_timestamp || null,
        last_seen_timestamp: r.last_seen_timestamp || null,
        evidence: r.evidence || {},
        metadata: r.metadata || {},
      };
    });

    const summary = raw.summary || {
      workers_count: entities.filter((e) => e.type === "PERSON").length,
      vehicles_count: entities.filter((e) => e.type === "VEHICLE").length,
      machines_count: entities.filter((e) => e.type === "MACHINE").length,
      objects_count: entities.filter((e) => e.type === "OBJECT").length,
      active_zones_count: entities.filter((e) => e.type === "ZONE").length,
      total_active_relationships: relationships.length,
    };

    return {
      ok: true,
      data: {
        timestamp: raw.timestamp || 0.0,
        frame_id: raw.frame_id || 0,
        camera_id: raw.camera_id || "cam_01",
        is_live: Boolean(raw.is_live),
        source_mode: raw.source_mode || (raw.is_live ? "LIVE" : "LATEST SCENE SNAPSHOT"),
        entities,
        relationships,
        summary,
        active_zones: raw.active_zones || [],
        active_behaviors: raw.active_behaviors || {},
      },
    };
  },
};
