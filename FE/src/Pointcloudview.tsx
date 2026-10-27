import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PointCloudData } from "./types";

export type ColorMode = "rgb" | "class";

type Props = {
  data: PointCloudData;
  hidden: Set<number>;
  colorMode: ColorMode;
};

type Layer = {
  points: THREE.Points;
  classMat: THREE.PointsMaterial;
  rgbMat: THREE.PointsMaterial | null;
};

// 軸の先端に文字（X, Y, Z）を表示するためのヘルパー関数
function createAxisLabel(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = color;
  ctx.font = "bold 48px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 32, 32);

  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.4, 0.4, 0.4);
  return sprite;
}

export function PointCloudView({ data, hidden, colorMode }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const layersRef = useRef<Map<number, Layer>>(new Map());

  // 3Dシーンの作成(データが変わったときだけ作り直す)
  useEffect(() => {
    const mount = mountRef.current!;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#14181c");

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 5000);
    camera.up.set(0, 0, 1); // 測量データはZが上

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    // --- 画面の左下に常駐する方位ギズモ用のシーンを作成 ---
    const gizmoScene = new THREE.Scene();
    const gizmoCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    gizmoCamera.up.set(0, 0, 1);
    
    const axesHelper = new THREE.AxesHelper(2);
    gizmoScene.add(axesHelper);

    // ★各軸の先端にX, Y, Zのラベルを追加（赤=X, 緑=Y, 青=Z）
    const labelX = createAxisLabel("X", "#ff3653");
    labelX.position.set(2.2, 0, 0);
    gizmoScene.add(labelX);

    const labelY = createAxisLabel("Y", "#009b0d");
    labelY.position.set(0, 2.2, 0);
    gizmoScene.add(labelY);

    const labelZ = createAxisLabel("Z", "#2575fc");
    labelZ.position.set(0, 0, 2.2);
    gizmoScene.add(labelZ);
    // ----------------------------------------------------

    // クラスごとに座標と色をまとめる
    const counts = new Map<number, number>();
    for (const c of data.classes) counts.set(c, (counts.get(c) ?? 0) + 1);
    const posBuf = new Map<number, Float32Array>();
    const colBuf = new Map<number, Uint8Array>();
    const cursor = new Map<number, number>();
    for (const [id, n] of counts) {
      posBuf.set(id, new Float32Array(n * 3));
      if (data.colors) colBuf.set(id, new Uint8Array(n * 3));
      cursor.set(id, 0);
    }
    const box = new THREE.Box3();
    const tmp = new THREE.Vector3();
    for (let i = 0; i < data.classes.length; i++) {
      const id = data.classes[i];
      const k = cursor.get(id)!;
      const pb = posBuf.get(id)!;
      const x = data.positions[i * 3];
      const y = data.positions[i * 3 + 1];
      const z = data.positions[i * 3 + 2];
      pb[k] = x;
      pb[k + 1] = y;
      pb[k + 2] = z;
      if (data.colors) {
        const cb = colBuf.get(id)!;
        cb[k] = data.colors[i * 3];
        cb[k + 1] = data.colors[i * 3 + 1];
        cb[k + 2] = data.colors[i * 3 + 2];
      }
      cursor.set(id, k + 3);
      box.expandByPoint(tmp.set(x, y, z));
    }

    const center = box.getCenter(new THREE.Vector3());
    const diag = Math.max(box.getSize(new THREE.Vector3()).length(), 1e-3);

    // メインシーンにグリッド（床面）を追加
    const gridHelper = new THREE.GridHelper(diag, 20, 0x888888, 0x444444);
    gridHelper.rotation.x = Math.PI / 2;
    gridHelper.position.set(center.x, center.y, box.min.z);
    scene.add(gridHelper);

    const pointSize = diag / 800;
    camera.near = diag / 1000;
    camera.far = diag * 20;
    camera.updateProjectionMatrix();

    const layers = new Map<number, Layer>();
    for (const [id, pb] of posBuf) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(pb, 3));
      const cb = colBuf.get(id);
      if (cb) geometry.setAttribute("color", new THREE.BufferAttribute(cb, 3, true));
      const color = data.classDefs.find((d) => d.id === id)?.color ?? "#ffffff";
      const classMat = new THREE.PointsMaterial({ size: pointSize, color });
      const rgbMat = cb ? new THREE.PointsMaterial({ size: pointSize, vertexColors: true }) : null;
      const points = new THREE.Points(geometry, classMat);
      scene.add(points);
      layers.set(id, { points, classMat, rgbMat });
    }
    layersRef.current = layers;

    camera.position.set(center.x + diag * 0.6, center.y - diag * 0.6, center.z + diag * 0.45);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.update();

    const resize = () => {
      const w = mount.clientWidth;
      const h = Math.max(mount.clientHeight, 1);
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);
    resize();

    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();

      // --- 1. メインの点群シーンを描画 ---
      renderer.setViewport(0, 0, mount.clientWidth, mount.clientHeight);
      renderer.setScissor(0, 0, mount.clientWidth, mount.clientHeight);
      renderer.setScissorTest(false);
      renderer.render(scene, camera);

      // --- 2. 画面の【左下】に方位ギズモを重ねて描画 ---
      const gizmoSize = 120; // ギズモの表示サイズ（ピクセル）
      const margin = 20;     // 画面端からの余白
      const gx = margin;                 // 左からの距離
      const gy = margin;                 // 下からの距離

      // メインカメラの回転をギズモカメラに同期
      const gizmoDist = 4;
      gizmoCamera.position.copy(camera.position).sub(controls.target).normalize().multiplyScalar(gizmoDist);
      gizmoCamera.up.copy(camera.up);
      gizmoCamera.lookAt(0, 0, 0);

      renderer.setViewport(gx, gy, gizmoSize, gizmoSize);
      renderer.setScissor(gx, gy, gizmoSize, gizmoSize);
      renderer.setScissorTest(true);
      
      renderer.autoClear = false;
      renderer.clearDepth();
      renderer.render(gizmoScene, gizmoCamera);
      renderer.autoClear = true;
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      axesHelper.dispose();
      gridHelper.dispose();
      // ラベルのテクスチャやマテリアルの破棄
      labelX.material.map?.dispose();
      labelX.material.dispose();
      labelY.material.map?.dispose();
      labelY.material.dispose();
      labelZ.material.map?.dispose();
      labelZ.material.dispose();

      for (const l of layers.values()) {
        l.points.geometry.dispose();
        l.classMat.dispose();
        l.rgbMat?.dispose();
      }
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [data]);

  // レイヤの表示・非表示と、色の切り替え
  useEffect(() => {
    for (const [id, l] of layersRef.current) {
      l.points.visible = !hidden.has(id);
      l.points.material = colorMode === "rgb" && l.rgbMat ? l.rgbMat : l.classMat;
    }
  }, [hidden, colorMode, data]);

  return <div className="viewport" ref={mountRef} aria-label="点群の3Dビュー" />;
}