import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PointCloudData } from "./types";

type Props = {
  data: PointCloudData;
  hidden: Set<number>; // 非表示にするクラス番号
};

export function PointCloudView({ data, hidden }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const layersRef = useRef<Map<number, THREE.Points>>(new Map());

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

    // クラスごとに座標をまとめる
    const counts = new Map<number, number>();
    for (const c of data.classes) counts.set(c, (counts.get(c) ?? 0) + 1);
    const buffers = new Map<number, Float32Array>();
    const cursor = new Map<number, number>();
    for (const [id, n] of counts) {
      buffers.set(id, new Float32Array(n * 3));
      cursor.set(id, 0);
    }
    const box = new THREE.Box3();
    const tmp = new THREE.Vector3();
    for (let i = 0; i < data.classes.length; i++) {
      const id = data.classes[i];
      const buf = buffers.get(id)!;
      const k = cursor.get(id)!;
      const x = data.positions[i * 3];
      const y = data.positions[i * 3 + 1];
      const z = data.positions[i * 3 + 2];
      buf[k] = x;
      buf[k + 1] = y;
      buf[k + 2] = z;
      cursor.set(id, k + 3);
      box.expandByPoint(tmp.set(x, y, z));
    }

    const layers = new Map<number, THREE.Points>();
    for (const [id, buf] of buffers) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(buf, 3));
      const color = data.classDefs.find((d) => d.id === id)?.color ?? "#ffffff";
      const material = new THREE.PointsMaterial({ size: 0.12, color });
      const points = new THREE.Points(geometry, material);
      scene.add(points);
      layers.set(id, points);
    }
    layersRef.current = layers;

    // カメラを点群の中心に向ける
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    camera.position.set(center.x + size * 0.6, center.y - size * 0.6, center.z + size * 0.45);
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
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      for (const p of layers.values()) {
        p.geometry.dispose();
        (p.material as THREE.Material).dispose();
      }
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [data]);

  // レイヤの表示・非表示の切り替え
  useEffect(() => {
    for (const [id, points] of layersRef.current) {
      points.visible = !hidden.has(id);
    }
  }, [hidden, data]);

  return <div className="viewport" ref={mountRef} aria-label="点群の3Dビュー" />;
}