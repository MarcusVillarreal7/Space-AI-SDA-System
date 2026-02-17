import { useRef, useEffect, useMemo, useCallback } from 'react';
import { Viewer, Entity, PolylineGraphics } from 'resium';
import {
  Viewer as CesiumViewer,
  Cartesian2,
  Cartesian3,
  Color,
  Ion,
  NearFarScalar,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  OpenStreetMapImageryProvider,
  defined,
  CallbackProperty,
  MaterialProperty,
  PolylineDashMaterialProperty,
} from 'cesium';
import { useSimStore } from '../store/useSimStore';
import { TIER_COLORS } from '../types';
import type { ThreatTier, ObjectType } from '../types';

// Disable Ion — we use dark Carto tiles (no auth needed)
Ion.defaultAccessToken = '';

function tierToColor(tier: ThreatTier): Color {
  const hex = TIER_COLORS[tier] || TIER_COLORS.MINIMAL;
  return Color.fromCssColorString(hex);
}

const PULSE_TIERS = new Set<ThreatTier>(['CRITICAL', 'ELEVATED']);

const TYPE_PIXEL_SIZE: Record<ObjectType, number> = {
  PAYLOAD: 4,
  DEBRIS: 2,
  ROCKET_BODY: 3,
};

export function Globe() {
  const viewerRef = useRef<CesiumViewer | null>(null);
  const handlerRef = useRef<ScreenSpaceEventHandler | null>(null);
  const imagerySetRef = useRef(false);
  const objects = useSimStore((s) => s.objects);
  const selectObject = useSimStore((s) => s.selectObject);
  const selectedObjectId = useSimStore((s) => s.selectedObjectId);
  const prediction = useSimStore((s) => s.selectedPrediction);
  const textFilter = useSimStore((s) => s.textFilter);
  const typeFilter = useSimStore((s) => s.typeFilter);

  // Build point data — apply TrackingTab filters so Globe stays in sync
  const pointData = useMemo(() => {
    let filtered = objects;
    if (typeFilter !== 'ALL') {
      filtered = filtered.filter((o) => o.object_type === typeFilter || o.id === selectedObjectId);
    }
    if (textFilter) {
      const lc = textFilter.toLowerCase();
      filtered = filtered.filter(
        (o) =>
          o.name.toLowerCase().includes(lc) ||
          String(o.id).includes(lc) ||
          o.id === selectedObjectId
      );
    }
    return filtered.map((obj) => {
      const shouldPulse = PULSE_TIERS.has(obj.threat_tier);
      const baseSize = TYPE_PIXEL_SIZE[obj.object_type] ?? 4;
      return {
        id: obj.id,
        name: obj.name,
        position: Cartesian3.fromDegrees(obj.lon, obj.lat, obj.alt_km * 1000),
        color: tierToColor(obj.threat_tier),
        tier: obj.threat_tier,
        pixelSize: obj.id === selectedObjectId ? 8 : shouldPulse ? 6 : baseSize,
        shouldPulse,
      };
    });
  }, [objects, selectedObjectId, textFilter, typeFilter]);

  // Build predicted trajectory polyline positions
  const predictionPositions = useMemo(() => {
    if (!prediction || prediction.object_id !== selectedObjectId || !prediction.points.length) {
      return null;
    }
    const coords: number[] = [];
    for (const pt of prediction.points) {
      coords.push(pt.lon, pt.lat, pt.alt_km * 1000);
    }
    return Cartesian3.fromDegreesArrayHeights(coords);
  }, [prediction, selectedObjectId]);

  // Set up click handler on the Cesium viewer
  const setupClickHandler = useCallback(
    (viewer: CesiumViewer) => {
      if (handlerRef.current) {
        handlerRef.current.destroy();
      }
      const handler = new ScreenSpaceEventHandler(viewer.scene.canvas);
      handler.setInputAction((movement: { position: Cartesian2 }) => {
        const picked = viewer.scene.pick(movement.position);
        if (defined(picked) && picked.id && picked.id.name) {
          // Entity names are satellite names — find the matching object
          const obj = objects.find((o) => o.name === picked.id.name);
          if (obj) {
            selectObject(obj.id === selectedObjectId ? null : obj.id);
            return;
          }
        }
        // Clicked empty space — deselect
        selectObject(null);
      }, ScreenSpaceEventType.LEFT_CLICK);
      handlerRef.current = handler;
    },
    [objects, selectObject, selectedObjectId]
  );

  // Attach click handler when viewer is ready
  useEffect(() => {
    if (viewerRef.current) {
      setupClickHandler(viewerRef.current);
    }
    return () => {
      if (handlerRef.current) {
        handlerRef.current.destroy();
        handlerRef.current = null;
      }
    };
  }, [setupClickHandler]);

  // Fly to selected object
  useEffect(() => {
    if (selectedObjectId === null || !viewerRef.current) return;
    const obj = objects.find((o) => o.id === selectedObjectId);
    if (!obj) return;

    viewerRef.current.camera.flyTo({
      destination: Cartesian3.fromDegrees(obj.lon, obj.lat, obj.alt_km * 1000 + 2000000),
      duration: 1.5,
    });
  }, [selectedObjectId]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex-1 relative">
      <Viewer
        full
        ref={(ref: any) => {
          if (ref?.cesiumElement) {
            const viewer = ref.cesiumElement as CesiumViewer;
            if (viewerRef.current !== viewer) {
              viewerRef.current = viewer;
              setupClickHandler(viewer);
              // Replace Ion imagery with dark Carto basemap tiles
              if (!imagerySetRef.current) {
                imagerySetRef.current = true;
                viewer.imageryLayers.removeAll();
                const provider = new OpenStreetMapImageryProvider({
                  url: 'https://basemaps.cartocdn.com/dark_all/',
                });
                viewer.imageryLayers.addImageryProvider(provider);
                // Dark sky background
                viewer.scene.backgroundColor = Color.fromCssColorString('#0a0e1a');
              }
            }
          }
        }}
        timeline={false}
        animation={false}
        homeButton={false}
        sceneModePicker={false}
        baseLayerPicker={false}
        geocoder={false}
        navigationHelpButton={false}
        fullscreenButton={false}
        selectionIndicator={false}
        infoBox={false}
        scene3DOnly
        requestRenderMode={false}
      >
        {pointData.map((pt) => (
          <Entity
            key={pt.id}
            position={pt.position}
            point={{
              pixelSize: pt.shouldPulse
                ? new CallbackProperty(() => {
                    const t = Date.now() / 500;
                    return pt.id === selectedObjectId ? 8 + Math.sin(t) * 3 : 5 + Math.sin(t) * 2;
                  }, false) as any
                : pt.pixelSize,
              color: pt.color,
              outlineColor: pt.id === selectedObjectId ? Color.WHITE : pt.shouldPulse ? pt.color.withAlpha(0.4) : Color.TRANSPARENT,
              outlineWidth: pt.id === selectedObjectId ? 2 : pt.shouldPulse ? 3 : 0,
              scaleByDistance: new NearFarScalar(1e6, 1.5, 1e8, 0.5),
            }}
            name={pt.name}
            description={`ID: ${pt.id} | Tier: ${pt.tier}`}
          />
        ))}

        {/* Predicted trajectory polyline for selected object */}
        {predictionPositions && (
          <Entity>
            <PolylineGraphics
              positions={predictionPositions}
              width={2}
              material={new PolylineDashMaterialProperty({
                color: Color.CYAN.withAlpha(0.7),
                dashLength: 16,
              }) as unknown as MaterialProperty}
            />
          </Entity>
        )}
      </Viewer>
    </div>
  );
}
