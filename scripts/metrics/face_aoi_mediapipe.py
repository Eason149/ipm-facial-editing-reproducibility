"""
face_aoi_mediapipe.py
=====================
修改：cy184 2026/5/28
修改：cy184 2026/5/29
批量对人脸图片运行 MediaPipe FaceMesh，自动生成六个AOI坐标：
    Eye, Nose, Mouth, Cheek_L, Cheek_R, Skin（多边形掩膜）

依赖：
    pip install mediapipe==0.10.14 opencv-python numpy

用法：
    python face_aoi_mediapipe.py --input_dir ./stimuli --output_dir ./aoi_output
    python face_aoi_mediapipe.py --input_dir ./stimuli --output_dir ./aoi_output --visualize
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import csv
import argparse
from pathlib import Path


# ─────────────────────────────────────────────
# AOI 定义函数
# ─────────────────────────────────────────────

def compute_aois(pts, img_w, img_h, pad=12):
    """
    根据 MediaPipe 468点地标，计算六个AOI的坐标。

    Parameters
    ----------
    pts  : dict  {landmark_index: (x, y)}  像素坐标
    pad  : int   各AOI矩形的外扩像素数

    Returns
    -------
    aois : dict  各AOI的坐标字典，Skin额外含掩膜
    """

    def clip_x(x): return max(0, min(img_w - 1, x))
    def clip_y(y): return max(0, min(img_h - 1, y))

    # ── Eye（左右合并，含眉）──────────────────────────────────
    eye_y1 = clip_y(min(pts[70][1],  pts[300][1]) - pad)   # 两侧眉毛最高点
    eye_y2 = clip_y(max(pts[145][1], pts[374][1]) + pad)   # 两侧眼下缘
    eye_tail_pad = max(pad, int(pad * 1.5))
    eye = {
        'x1': clip_x(min(pts[33][0], pts[130][0], pts[226][0]) - eye_tail_pad),
        'y1': eye_y1,
        'x2': clip_x(max(pts[263][0], pts[359][0], pts[446][0]) + eye_tail_pad),
        'y2': eye_y2,
    }

    # ── Nose（上边界紧接Eye下边界，不重叠）──────────────────
    nose = {
        'x1': clip_x(pts[64][0]  - pad),   # 左鼻翼外缘
        'y1': clip_y(eye_y2 + 1),           # 紧接Eye下边界
        'x2': clip_x(pts[294][0] + pad),   # 右鼻翼外缘
        'y2': clip_y(pts[4][1]   + pad),   # 鼻尖
    }

    # ── Mouth ────────────────────────────────────────────────
    mouth = {
        'x1': clip_x(pts[61][0]  - pad),   # 左嘴角
        'y1': clip_y(pts[0][1]   - pad),   # 上唇峰
        'x2': clip_x(pts[291][0] + pad),   # 右嘴角
        'y2': clip_y(pts[17][1]  + pad),   # 下唇底
    }

    # ── Skin（脸部轮廓多边形内，排除其余AOI，从发际线开始）──
    face_contour_lm = [
        10,  338, 297, 332, 284, 251, 389, 356, 454,
        323, 361, 288, 397, 365, 379, 378, 400, 377,
        152,
        148, 176, 149, 150, 136, 172,  58, 132,  93,
        234, 127, 162,  21,  54, 103,  67, 109,  10,
    ]
    face_poly = np.array([pts[i] for i in face_contour_lm], dtype=np.int32)

    face_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_poly], 255)

    def rect_mask(rect):
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.rectangle(mask, (rect['x1'], rect['y1']), (rect['x2'], rect['y2']), 255, -1)
        return mask

    eye_mask   = rect_mask(eye)
    nose_mask  = rect_mask(nose)
    mouth_mask = rect_mask(mouth)

    def make_cheek(name, poly_points):
        poly = np.array(poly_points, dtype=np.int32)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        mask = cv2.bitwise_and(mask, face_mask)
        for exclude in [eye_mask, nose_mask, mouth_mask]:
            mask[exclude > 0] = 0
        return {
            'type': 'polygon_mask',
            'points': [{'x': int(x), 'y': int(y)} for x, y in poly_points],
            'note': f'{name}为脸颊多边形掩膜，已裁剪到脸部轮廓内并排除Eye/Nose/Mouth',
            'mask': mask,
        }

    # ── Cheek_L / Cheek_R ───────────────────────────────────
    # 使用多边形覆盖瘦脸主要改变的侧脸带状区域：外边界贴近原脸轮廓，
    # 内边界模拟瘦脸后的内缩轮廓。顶部回到眼尾/外眼下附近，
    # 但不横向吃进鼻旁和眼下中心区。
    cheek_top_l = clip_y(eye_y2 + max(2, pad // 3))
    cheek_top_r = clip_y(eye_y2 + max(2, pad // 3))
    cheek_lower_pad = max(8, int(pad * 0.75))

    cheek_inner_top_l = clip_x(min(pts[117][0], pts[50][0] + pad))
    cheek_inner_mid_l = clip_x(max(pts[187][0], min(pts[205][0],
                                                    cheek_inner_top_l + int(pad * 1.6))))
    cheek_inner_low_l = clip_x(min(mouth['x1'] - int(pad * 1.4),
                                   cheek_inner_mid_l + int(pad * 1.2)))

    cheek_inner_top_r = clip_x(max(pts[346][0], pts[280][0] - pad))
    cheek_inner_mid_r = clip_x(min(pts[411][0], max(pts[425][0],
                                                    cheek_inner_top_r - int(pad * 1.6))))
    cheek_inner_low_r = clip_x(max(mouth['x2'] + int(pad * 1.4),
                                   cheek_inner_mid_r - int(pad * 1.2)))

    cheek_l_points = [
        (clip_x(pts[234][0] + 2), cheek_top_l),
        (cheek_inner_top_l, cheek_top_l),
        (cheek_inner_mid_l, clip_y(nose['y2'] + cheek_lower_pad)),
        (cheek_inner_low_l, clip_y(mouth['y2'] + cheek_lower_pad)),
        (clip_x(pts[172][0]), clip_y(pts[172][1] + cheek_lower_pad)),
        (clip_x(pts[132][0]), clip_y(pts[132][1])),
        (clip_x(pts[137][0]), clip_y(pts[137][1])),
    ]
    cheek_r_points = [
        (cheek_inner_top_r, cheek_top_r),
        (clip_x(pts[454][0] - 2), cheek_top_r),
        (clip_x(pts[366][0]), clip_y(pts[366][1])),
        (clip_x(pts[361][0]), clip_y(pts[361][1])),
        (clip_x(pts[397][0]), clip_y(pts[397][1] + cheek_lower_pad)),
        (cheek_inner_low_r, clip_y(mouth['y2'] + cheek_lower_pad)),
        (cheek_inner_mid_r, clip_y(nose['y2'] + cheek_lower_pad)),
    ]
    cheek_l = make_cheek('Cheek_L', cheek_l_points)
    cheek_r = make_cheek('Cheek_R', cheek_r_points)

    skin_mask = face_mask.copy()
    for r in [eye, nose, mouth]:
        cv2.rectangle(skin_mask, (r['x1'], r['y1']), (r['x2'], r['y2']), 0, -1)
    for cheek in [cheek_l, cheek_r]:
        skin_mask[cheek['mask'] > 0] = 0
    skin_mask[:pts[10][1], :] = 0   # 发际线以上裁除

    aois = {
        'Eye':     eye,
        'Nose':    nose,
        'Mouth':   mouth,
        'Cheek_L': cheek_l,
        'Cheek_R': cheek_r,
        'Skin': {
            'type':                   'polygon_mask',
            'face_contour_landmarks': face_contour_lm,
            'face_contour_points': [
                {'landmark': i, 'x': int(pts[i][0]), 'y': int(pts[i][1])}
                for i in face_contour_lm
            ],
            'top_boundary_landmark':  10,
            'top_boundary_point': {
                'x': int(pts[10][0]),
                'y': int(pts[10][1]),
            },
            'note':                   '脸部轮廓多边形内排除其余五个AOI，发际线以上裁除',
            'mask':                   skin_mask,   # numpy array，不写入JSON
        }
    }
    return aois


# ─────────────────────────────────────────────
# 可视化函数
# ─────────────────────────────────────────────

COLORS_BGR = {
    'Eye':     (29,  158,  86),
    'Nose':    (213, 100, 127),
    'Mouth':   (48,   90, 216),
    'Cheek_L': (200, 130,  50),
    'Cheek_R': (200, 130,  50),
    'Skin':    (60,  180, 220),
}

KEY_LANDMARKS = [
    33, 130, 226, 263, 359, 446,   # 眼尾/眼角
    70, 300, 145, 374,             # 眉/眼下缘
    64, 294, 4, 6,                  # 鼻
    61, 291, 0, 17,                 # 嘴
    234, 137, 132, 172, 50, 117, 187, 205,  # 左脸颊多边形
    454, 366, 361, 397, 280, 346, 425, 411, # 右脸颊多边形
    10, 152,                        # 发际/下巴
]

def visualize_aois(img, aois, pts):
    vis     = img.copy()
    overlay = img.copy()

    # Skin 半透明填充（掩膜方式）
    skin_mask = aois['Skin']['mask']
    sc = np.array(COLORS_BGR['Skin'])
    overlay[skin_mask > 0] = (overlay[skin_mask > 0] * 0.65 + sc * 0.35).astype(np.uint8)

    # 其余 AOI 填充：Eye/Nose/Mouth 为矩形，Cheek 为多边形掩膜。
    for name in ['Eye', 'Nose', 'Mouth', 'Cheek_L', 'Cheek_R']:
        aoi = aois[name]
        c   = COLORS_BGR[name]
        if 'mask' in aoi:
            overlay[aoi['mask'] > 0] = c
        else:
            cv2.rectangle(overlay, (aoi['x1'], aoi['y1']), (aoi['x2'], aoi['y2']), c, -1)

    cv2.addWeighted(overlay, 0.30, vis, 0.70, 0, vis)

    # 边框 + 标签
    for name in ['Eye', 'Nose', 'Mouth', 'Cheek_L', 'Cheek_R']:
        aoi = aois[name]
        c   = COLORS_BGR[name]
        if 'mask' in aoi:
            contours, _ = cv2.findContours(aoi['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, c, 2)
            bbox, _ = skin_bbox_and_area(aoi['mask'])
        else:
            cv2.rectangle(vis, (aoi['x1'], aoi['y1']), (aoi['x2'], aoi['y2']), c, 2)
            bbox = aoi

        label_xy = (bbox['x1'] + 4, bbox['y1'] + 16)
        cv2.putText(vis, name, label_xy,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(vis, name, label_xy,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    # Skin 轮廓线 + 标签
    face_poly = np.array([pts[i] for i in aois['Skin']['face_contour_landmarks']], dtype=np.int32)
    cv2.polylines(vis, [face_poly], True, COLORS_BGR['Skin'], 1)
    hairline = pts[aois['Skin']['top_boundary_landmark']]
    cv2.putText(vis, 'Skin', (hairline[0] - 20, hairline[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(vis, 'Skin', (hairline[0] - 20, hairline[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS_BGR['Skin'], 1)

    # 关键地标点
    for idx in KEY_LANDMARKS:
        cv2.circle(vis, pts[idx], 4, (255, 255, 255), -1)
        cv2.circle(vis, pts[idx], 4, (0, 0, 0), 1)

    return vis


def make_partition_image(img_shape, aois, pts):
    """
    生成一张纯 AOI 分区图：白色背景，AOI 使用固定颜色填充。
    这张图比叠加图更适合检查各区域是否互相覆盖。
    """
    h, w = img_shape[:2]
    part = np.full((h, w, 3), 255, dtype=np.uint8)

    skin_mask = aois['Skin']['mask']
    part[skin_mask > 0] = COLORS_BGR['Skin']

    for name in ['Eye', 'Nose', 'Mouth', 'Cheek_L', 'Cheek_R']:
        aoi = aois[name]
        if 'mask' in aoi:
            part[aoi['mask'] > 0] = COLORS_BGR[name]
        else:
            cv2.rectangle(part, (aoi['x1'], aoi['y1']), (aoi['x2'], aoi['y2']),
                          COLORS_BGR[name], -1)

    face_poly = np.array([pts[i] for i in aois['Skin']['face_contour_landmarks']], dtype=np.int32)
    cv2.polylines(part, [face_poly], True, (70, 70, 70), 1)

    for name in ['Eye', 'Nose', 'Mouth', 'Cheek_L', 'Cheek_R']:
        aoi = aois[name]
        if 'mask' in aoi:
            contours, _ = cv2.findContours(aoi['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(part, contours, -1, (30, 30, 30), 1)
            bbox, _ = skin_bbox_and_area(aoi['mask'])
        else:
            cv2.rectangle(part, (aoi['x1'], aoi['y1']), (aoi['x2'], aoi['y2']),
                          (30, 30, 30), 1)
            bbox = aoi

        label_xy = (bbox['x1'] + 4, bbox['y1'] + 16)
        cv2.putText(part, name, label_xy,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(part, name, label_xy,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)

    hairline = pts[aois['Skin']['top_boundary_landmark']]
    cv2.putText(part, 'Skin', (hairline[0] - 20, hairline[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(part, 'Skin', (hairline[0] - 20, hairline[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)

    return part


def rect_area(aoi):
    return int(max(0, aoi['x2'] - aoi['x1'] + 1) *
               max(0, aoi['y2'] - aoi['y1'] + 1))


def skin_bbox_and_area(mask):
    area = int(np.count_nonzero(mask))
    if area == 0:
        return None, 0

    ys, xs = np.where(mask > 0)
    bbox = {
        'x1': int(xs.min()),
        'y1': int(ys.min()),
        'x2': int(xs.max()),
        'y2': int(ys.max()),
    }
    return bbox, area


def serialize_result(img_path, img_w, img_h, aois):
    serial_aois = {}
    for name in ['Eye', 'Nose', 'Mouth', 'Cheek_L', 'Cheek_R']:
        src = aois[name]
        if 'mask' in src:
            bbox, area = skin_bbox_and_area(src['mask'])
            aoi = {
                'type': src.get('type', 'polygon_mask'),
                'bbox': bbox,
                'x1': bbox['x1'] if bbox else None,
                'y1': bbox['y1'] if bbox else None,
                'x2': bbox['x2'] if bbox else None,
                'y2': bbox['y2'] if bbox else None,
                'area_px': area,
                'points': src.get('points', []),
                'note': src.get('note', ''),
            }
        else:
            aoi = {k: int(v) for k, v in src.items()}
            aoi['type'] = 'rectangle'
            aoi['area_px'] = rect_area(aoi)
        serial_aois[name] = aoi

    skin = {k: v for k, v in aois['Skin'].items() if k != 'mask'}
    skin_bbox, skin_area = skin_bbox_and_area(aois['Skin']['mask'])
    skin['bbox'] = skin_bbox
    skin['area_px'] = skin_area

    return {
        'image':      img_path.name,
        'image_size': {'width': int(img_w), 'height': int(img_h)},
        'aois':       serial_aois,
        'skin':       skin,
    }


def write_coordinates_csv(results, csv_path):
    fieldnames = [
        'image', 'image_width', 'image_height', 'aoi', 'type',
        'x1', 'y1', 'x2', 'y2', 'area_px',
        'points', 'face_contour_landmarks', 'face_contour_points', 'note',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            width = result['image_size']['width']
            height = result['image_size']['height']

            for name, aoi in result['aois'].items():
                writer.writerow({
                    'image': result['image'],
                    'image_width': width,
                    'image_height': height,
                    'aoi': name,
                    'type': aoi['type'],
                    'x1': aoi['x1'],
                    'y1': aoi['y1'],
                    'x2': aoi['x2'],
                    'y2': aoi['y2'],
                    'area_px': aoi['area_px'],
                    'points': json.dumps(aoi.get('points', []), ensure_ascii=False),
                    'face_contour_landmarks': '',
                    'face_contour_points': '',
                    'note': aoi.get('note', ''),
                })

            skin = result['skin']
            bbox = skin.get('bbox') or {'x1': '', 'y1': '', 'x2': '', 'y2': ''}
            writer.writerow({
                'image': result['image'],
                'image_width': width,
                'image_height': height,
                'aoi': 'Skin',
                'type': skin['type'],
                'x1': bbox['x1'],
                'y1': bbox['y1'],
                'x2': bbox['x2'],
                'y2': bbox['y2'],
                'area_px': skin['area_px'],
                'points': '',
                'face_contour_landmarks': ';'.join(map(str, skin['face_contour_landmarks'])),
                'face_contour_points': json.dumps(skin['face_contour_points'], ensure_ascii=False),
                'note': skin.get('note', ''),
            })


# ─────────────────────────────────────────────
# 单张图片处理
# ─────────────────────────────────────────────

def process_image(img_path, face_mesh, pad=12, visualize=False, output_dir=None):
    """
    对单张图片运行 MediaPipe，返回 AOI 坐标字典。
    如果 visualize=True，将叠加图和纯 AOI 分区图保存到 output_dir。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] 无法读取: {img_path}")
        return None

    h, w = img.shape[:2]
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res  = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        print(f"  [WARN] 未检测到人脸: {img_path.name}")
        return None

    lm  = res.multi_face_landmarks[0].landmark
    pts = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))}

    aois = compute_aois(pts, w, h, pad=pad)

    # 可视化
    if visualize and output_dir:
        vis_img = visualize_aois(img, aois, pts)
        vis_path = output_dir / (img_path.stem + '_aoi.jpg')
        cv2.imwrite(str(vis_path), vis_img)

        partition_img = make_partition_image(img.shape, aois, pts)
        partition_path = output_dir / (img_path.stem + '_partition.png')
        cv2.imwrite(str(partition_path), partition_img)

    return serialize_result(img_path, w, h, aois)


# ─────────────────────────────────────────────
# 批量处理
# ─────────────────────────────────────────────

def batch_process(input_dir, output_dir, pad=12, visualize=False):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinate_dir = output_dir / 'coordinates'
    coordinate_dir.mkdir(parents=True, exist_ok=True)

    exts       = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_paths  = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])
    print(f"找到 {len(img_paths)} 张图片，开始处理…\n")

    mp_face_mesh = mp.solutions.face_mesh
    all_results  = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for i, img_path in enumerate(img_paths, 1):
            print(f"[{i:3d}/{len(img_paths)}] {img_path.name}")
            result = process_image(img_path, face_mesh,
                                   pad=pad,
                                   visualize=visualize,
                                   output_dir=output_dir)
            if result:
                all_results.append(result)
                one_json = coordinate_dir / (img_path.stem + '_aoi.json')
                with open(one_json, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

    # 保存汇总 JSON
    summary_path = output_dir / 'aoi_all_images.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 保存长表 CSV：每张图片 x 每个 AOI 一行
    csv_path = output_dir / 'aoi_coordinates_long.csv'
    write_coordinates_csv(all_results, csv_path)

    print(f"\n完成！AOI坐标已保存到: {summary_path}")
    print(f"逐图坐标 JSON 已保存到: {coordinate_dir}")
    print(f"长表 CSV 已保存到: {csv_path}")
    print(f"成功处理: {len(all_results)} / {len(img_paths)} 张")
    return all_results


# ─────────────────────────────────────────────
# 主程序入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量提取人脸AOI坐标（MediaPipe FaceMesh）')
    parser.add_argument('--input_dir',  required=True,  help='刺激图片文件夹路径')
    parser.add_argument('--output_dir', required=True,  help='输出文件夹路径')
    parser.add_argument('--pad',        type=int, default=12, help='AOI外扩像素数（默认12）')
    parser.add_argument('--visualize',  action='store_true',  help='是否输出可视化图片')
    args = parser.parse_args()

    batch_process(
        input_dir  = args.input_dir,
        output_dir = args.output_dir,
        pad        = args.pad,
        visualize  = args.visualize,
    )
