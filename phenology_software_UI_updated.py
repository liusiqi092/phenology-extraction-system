# -*- coding: utf-8 -*-
# 植被物候提取桌面程序

import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from PySide6 import QtCore, QtWidgets


NODATA = -9999.0
MIN_YEAR = 1900
MAX_YEAR = 2100
PERIOD_COUNT = 23


def natural_key(text):
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split('(\\d+)', str(text))
    ]


def is_tif(path):
    return str(path).lower().endswith(('.tif', '.tiff'))


def parse_year_period_filename(path, index_type):
    filename = Path(path).stem.upper()
    index_type = index_type.upper()
    if index_type not in ('EVI', 'NDVI'):
        return None
    rules = [
        f'^(?P<year>19\\d{{2}}|20\\d{{2}})[_-]'
        f'(?P<period>\\d{{1,2}})[_-]{index_type}(?:[_-].*)?$',
        f'^{index_type}[_-](?P<year>19\\d{{2}}|20\\d{{2}})[_-]'
        f'(?P<period>\\d{{1,2}})(?:[_-].*)?$',
    ]
    for rule in rules:
        match = re.match(rule, filename, re.IGNORECASE)
        if match:
            found_year = int(match.group('year'))
            found_period = int(match.group('period'))
            if 1 <= found_period <= PERIOD_COUNT:
                return (found_year, found_period)
    return None

# 文件名必须完整对应到一年中的 23 个期次。
def find_year_files(input_folder, year, index_type):
    folder = Path(input_folder)
    if not folder.exists():
        return {
            'ok': False,
            'files': [],
            'missing': list(range(1, 24)),
            'duplicates': {},
            'message': '输入目录不存在：{}'.format(folder),
        }
    candidates = sorted(
        (item for item in folder.iterdir() if item.is_file() and is_tif(item)),
        key=lambda item: natural_key(item.name),
    )
    chosen = {}
    duplicates = {}
    # print("[debug] candidate rasters:", len(candidates))
    # legacy_files = sorted(folder.glob("*.tif"))  # 旧版未校验期次
    target_year = int(year)
    for item in candidates:
        parsed = parse_year_period_filename(item, index_type)
        if not parsed or parsed[0] != target_year:
            continue
        slot = parsed[1]
        if slot in chosen:
            duplicates.setdefault(slot, [chosen[slot]]).append(str(item))
        else:
            chosen[slot] = str(item)
    missing_slots = [
        slot for slot in range(1, PERIOD_COUNT + 1) if slot not in chosen
    ]
    if duplicates:
        return {
            'ok': False,
            'files': [],
            'missing': missing_slots,
            'duplicates': duplicates,
            'message': '{}年存在重复期数，无法确定正确数据。'.format(year),
        }
    if missing_slots:
        return {
            'ok': False,
            'files': [],
            'missing': missing_slots,
            'duplicates': {},
            'message': '{}年缺少{}期数据。'.format(year, len(missing_slots)),
        }
    ordered = [chosen[slot] for slot in range(1, PERIOD_COUNT + 1)]
    return {
        'ok': True,
        'files': ordered,
        'missing': [],
        'duplicates': {},
        'message': '{}年数据正常，共{}期。'.format(year, len(ordered)),
    }


def read_raster_scaled(path):
    with rasterio.open(path) as dataset:
        grid = dataset.read(1).astype(np.float32)
        profile = dataset.profile.copy()
        transform = dataset.transform
        crs = dataset.crs
        nodata = dataset.nodata
        if nodata is not None:
            grid[grid == nodata] = np.nan
        finite = np.isfinite(grid)
        if np.any(finite):
            peak = np.nanmax(grid)
            if peak > 1.5:
                grid = grid * 0.0001
        grid[(grid < -1.0) | (grid > 1.0)] = np.nan
        return (grid, profile, transform, crs)


def check_raster_consistency(files):
    if not files:
        raise ValueError('没有输入栅格。')
    with rasterio.open(files[0]) as dataset:
        shape = (dataset.height, dataset.width)
        transform = dataset.transform
        crs = dataset.crs
    for raster_path in files[1:]:
        with rasterio.open(raster_path) as dataset:
            if (dataset.height, dataset.width) != shape:
                raise ValueError('栅格尺寸不一致：\n{}'.format(raster_path))
            if dataset.transform != transform:
                raise ValueError('栅格空间变换参数不一致：\n{}'.format(raster_path))
            if dataset.crs != crs:
                raise ValueError('栅格坐标系不一致：\n{}'.format(raster_path))
    return (shape, transform, crs)


class RegionManager:

    def __init__(self):
        self.gdf = None
        self.shp_path = None
        self.fields = []
        self.current_field = None
        self.current_value = None

    def load_shp(self, shp_path):
        shp_path = str(shp_path)
        if not os.path.exists(shp_path):
            raise ValueError('SHP文件不存在：\n{}'.format(shp_path))
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            raise ValueError('SHP属性表为空。')
        if gdf.crs is None:
            raise ValueError('SHP没有坐标系定义。\n请先为SHP定义正确坐标系。')
        self.gdf = gdf
        self.shp_path = shp_path
        self.fields = [c for c in gdf.columns if c != 'geometry']
        return self.fields

    def get_field_values(self, field):
        if self.gdf is None:
            return []
        if field not in self.gdf.columns:
            return []
        values = []
        for v in self.gdf[field].dropna().unique():
            text = str(v).strip()
            if text != '':
                values.append(text)
        values = sorted(list(set(values)), key=natural_key)
        return values

    def get_selected_geometries(self, mode, field=None, value=None):
        if self.gdf is None:
            raise ValueError('请先加载SHP。')
        gdf = self.gdf.copy()
        if mode == '全域':
            selected = gdf
        else:
            if field is None:
                raise ValueError('请选择属性字段。')
            if value is None:
                raise ValueError('请选择区域。')
            selected = gdf[gdf[field].astype(str).str.strip() == str(value).strip()]
        if selected.empty:
            raise ValueError('没有找到对应的SHP面。')
        geometries = []
        for geom in selected.geometry:
            if geom is None:
                continue
            if geom.is_empty:
                continue
            if not geom.is_valid:
                try:
                    geom = geom.buffer(0)
                except Exception:
                    continue
            if geom is None or geom.is_empty:
                continue
            geometries.append(geom)
        if not geometries:
            raise ValueError('所选区域没有有效几何面。')
        return geometries


def rasterize_region(shp_path, mode, field, value, reference_raster, output_mask):
    regions = RegionManager()
    regions.load_shp(shp_path)
    with rasterio.open(reference_raster) as reference:
        shape = (reference.height, reference.width)
        transform = reference.transform
        raster_crs = reference.crs
        profile = reference.profile.copy()
    if raster_crs is None:
        raise ValueError('输入EVI/NDVI栅格没有坐标系定义，无法按SHP边界生成掩膜。')
    if regions.gdf.crs != raster_crs:
        regions.gdf = regions.gdf.to_crs(raster_crs)
    selected_shapes = regions.get_selected_geometries(mode, field, value)
    region_grid = rasterize(
        [(feature_geom, 1) for feature_geom in selected_shapes],
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype='uint8',
        all_touched=False,
    )
    if not np.any(region_grid):
        raise ValueError('当前SHP区域与输入EVI/NDVI栅格没有重叠像元。\n请检查SHP坐标系、空间位置以及所选属性区域。')
    profile.update(dtype='uint8', count=1, nodata=0, compress='lzw')
    output_mask = str(output_mask)
    os.makedirs(os.path.dirname(output_mask), exist_ok=True)
    with rasterio.open(output_mask, 'w', **profile) as output:
        output.write(region_grid, 1)
    return region_grid.astype(bool)


def fill_nan_series(y):
    values = np.asarray(y, dtype=np.float32)
    good = np.isfinite(values)
    good_count = int(np.count_nonzero(good))
    if good_count == 0:
        return None
    filled = values.copy()
    if good_count == 1:
        filled.fill(filled[good][0])
        return filled
    positions = np.arange(filled.size)
    filled[~good] = np.interp(positions[~good], positions[good], filled[good])
    return filled


def sg_filter_series(y, window=7, polyorder=2, missing_method='线性插值'):
    y = np.asarray(y, dtype=np.float32)
    if missing_method == '线性插值':
        filled = fill_nan_series(y)
    else:
        filled = fill_nan_series(y)
    if filled is None:
        return None
    n = len(filled)
    window = int(window)
    polyorder = int(polyorder)
    if window >= n:
        window = n if n % 2 == 1 else n - 1
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return filled
    if polyorder >= window:
        polyorder = window - 1
    if polyorder < 1:
        polyorder = 1
    try:
        result = savgol_filter(filled, window_length=window, polyorder=polyorder, mode='interp')
        return result.astype(np.float32)
    except Exception:
        return filled


def build_hants_matrix(n, harmonics, use_trend):
    samples = np.arange(n, dtype=float)
    basis = [np.ones(n)]
    if use_trend:
        basis.append(samples / max(float(n - 1), 1.0))
    for order in range(1, harmonics + 1):
        phase = 2.0 * np.pi * order * samples / n
        basis.extend((np.cos(phase), np.sin(phase)))
    return np.column_stack(basis)


def hants_series(
    y,
    harmonics=2,
    max_iterations=500,
    outlier_threshold=0.05,
    outlier_direction='双向异常值',
    min_valid=8,
    trend='不使用',
    weight_mode='等权',
):
    y = np.asarray(y, dtype=np.float32)
    observed = np.isfinite(y)
    if observed.sum() < int(min_valid):
        return None
    filled = fill_nan_series(y)
    if filled is None:
        return None
    n = len(filled)
    harmonics = int(harmonics)
    harmonics = max(1, min(harmonics, max(1, (n - 1) // 2)))
    max_iterations = max(1, int(max_iterations))
    min_valid = max(3, int(min_valid))
    outlier_threshold = max(0.0001, float(outlier_threshold))
    use_trend = trend == '一阶趋势'
    design = build_hants_matrix(n, harmonics, use_trend)
    weights = np.ones(n, dtype=np.float64)
    weights[~observed] = 0.0
    modelled = filled.astype(np.float64)
    last_outliers = None
    for iteration in range(max_iterations):
        effective_count = int(np.sum(weights > 0))
        if effective_count < min_valid:
            break
        try:
            if weight_mode == '鲁棒权重':
                sqrt_w = np.sqrt(np.maximum(weights, 0.0))
                weighted_design = design * sqrt_w[:, None]
                weighted_signal = filled * sqrt_w
                coeffs, _, _, _ = np.linalg.lstsq(weighted_design, weighted_signal, rcond=None)
            else:
                active = weights > 0
                coeffs, _, _, _ = np.linalg.lstsq(design[active], filled[active], rcond=None)
            modelled = design @ coeffs
        except Exception:
            return filled
        errors = filled - modelled
        error_size = np.abs(errors)
        outliers = np.zeros(n, dtype=bool)
        reject = observed.copy()
        if outlier_direction == '仅高异常值':
            outliers = errors > outlier_threshold
        elif outlier_direction == '仅低异常值':
            outliers = errors < -outlier_threshold
        else:
            outliers = error_size > outlier_threshold
        outliers &= reject
        if weight_mode == '鲁棒权重':
            weights = np.ones(n, dtype=np.float64)
            weights[~observed] = 0.0
            for i in range(n):
                if not observed[i]:
                    continue
                ar = error_size[i]
                if ar <= outlier_threshold:
                    weights[i] = 1.0
                else:
                    weights[i] = outlier_threshold / max(ar, 1e-08)
            weights[outliers] = 0.0
        else:
            next_weights = weights.copy()
            next_weights[outliers] = 0.0
            weights = next_weights
        if last_outliers is not None:
            if np.array_equal(outliers, last_outliers):
                break
        last_outliers = outliers.copy()
        if np.sum(weights > 0) < min_valid:
            break
    try:
        active = weights > 0
        if np.sum(active) >= min_valid:
            if weight_mode == '鲁棒权重':
                sqrt_w = np.sqrt(np.maximum(weights, 0.0))
                weighted_design = design * sqrt_w[:, None]
                weighted_signal = filled * sqrt_w
                coeffs, _, _, _ = np.linalg.lstsq(weighted_design, weighted_signal, rcond=None)
            else:
                coeffs, _, _, _ = np.linalg.lstsq(design[active], filled[active], rcond=None)
            result = design @ coeffs
        else:
            result = modelled
        return result.astype(np.float32)
    except Exception:
        return modelled.astype(np.float32)


def double_logistic_standard(t, a, b, c, d, e, f):
    t = np.asarray(t, dtype=float)
    e = max(float(e), 1e-06)
    return a + b / (1.0 + np.exp(np.clip(-(t - c) / e, -50, 50))) - d / (1.0 + np.exp(np.clip(-(t - f) / e, -50, 50)))


def double_logistic_asymmetric(t, a, b, c, d, e, f, g):
    t = np.asarray(t, dtype=float)
    e = max(float(e), 1e-06)
    g = max(float(g), 1e-06)
    return a + b / (1.0 + np.exp(np.clip(-(t - c) / e, -50, 50))) - d / (1.0 + np.exp(np.clip(-(t - f) / g, -50, 50)))


def get_double_logistic_initial(filled, model_type, initial_mode):
    n = len(filled)
    minimum = float(np.min(filled))
    maximum = float(np.max(filled))
    amplitude = max(maximum - minimum, 0.01)
    peak = int(np.argmax(filled))
    if initial_mode == '自动估计':
        c = max(0.0, min(float(peak) * 0.7, float(n - 1)))
        f = min(float(n - 1), max(float(peak) + max(2.0, n * 0.2), n * 0.7))
        e = max(1.0, n * 0.08)
        g = max(1.0, n * 0.08)
        if model_type == '非对称双逻辑斯蒂函数':
            return [minimum, amplitude, c, amplitude, e, f, g]
        return [minimum, amplitude, c, amplitude, e, f]
    c = n * 0.25
    f = n * 0.75
    e = 3.0
    g = 3.0
    if model_type == '非对称双逻辑斯蒂函数':
        return [minimum, amplitude, c, amplitude, e, f, g]
    return [minimum, amplitude, c, amplitude, e, f]


def double_logistic_fit(
    y,
    model_type='标准双逻辑斯蒂函数',
    initial_mode='自动估计',
    max_iterations=500,
    tolerance=1e-05,
    outlier_method='残差法',
    min_valid=8,
    weight_mode='等权',
):
    y = np.asarray(y, dtype=np.float32)
    valid = np.isfinite(y)
    if valid.sum() < int(min_valid):
        return None
    filled = fill_nan_series(y)
    if filled is None:
        return None
    n = len(filled)
    t = np.arange(n, dtype=float)
    max_iterations = max(100, int(max_iterations))
    tolerance = max(1e-10, float(tolerance))
    min_valid = max(5, int(min_valid))
    if model_type == '非对称双逻辑斯蒂函数':
        func = double_logistic_asymmetric
        p0 = get_double_logistic_initial(filled, model_type, initial_mode)
        lower = [-1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1]
        upper = [1.0, 2.0, float(n - 1), 2.0, 20.0, float(n - 1), 20.0]
    else:
        func = double_logistic_standard
        p0 = get_double_logistic_initial(filled, model_type, initial_mode)
        lower = [-1.0, 0.0, 0.0, 0.0, 0.1, 0.0]
        upper = [1.0, 2.0, float(n - 1), 2.0, 20.0, float(n - 1)]
    p0 = np.asarray(p0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    p0 = np.minimum(np.maximum(p0, lower + 1e-06), upper - 1e-06)
    current_valid = valid.copy()
    popt = None
    if outlier_method == '不处理':
        iteration_count = 1
    else:
        iteration_count = min(10, max(1, int(max_iterations // 100)))
    for iteration in range(iteration_count):
        if current_valid.sum() < min_valid:
            break
        x = t[current_valid]
        yy = filled[current_valid]
        try:
            if weight_mode == '鲁棒权重':
                popt, _ = curve_fit(func, x, yy, p0=p0, bounds=(lower, upper), maxfev=max_iterations)
            else:
                popt, _ = curve_fit(func, x, yy, p0=p0, bounds=(lower, upper), maxfev=max_iterations)
        except Exception:
            return filled.astype(np.float32)
        fitted_all = func(t, *popt)
        residual = filled - fitted_all
        if outlier_method == '残差法':
            threshold = max(0.01, float(np.nanstd(residual[current_valid])) * 2.0)
            bad = np.abs(residual) > threshold
            bad &= current_valid
        elif outlier_method == 'IQR法':
            residual_valid = residual[current_valid]
            q1 = np.percentile(residual_valid, 25)
            q3 = np.percentile(residual_valid, 75)
            iqr = q3 - q1
            lower_r = q1 - 1.5 * iqr
            upper_r = q3 + 1.5 * iqr
            bad = (residual < lower_r) | (residual > upper_r)
            bad &= current_valid
        else:
            bad = np.zeros(n, dtype=bool)
        if weight_mode == '鲁棒权重':
            sigma = np.std(residual[current_valid])
            sigma = max(sigma, 0.0001)
            weights = np.ones(n, dtype=float)
            weights[current_valid] = 1.0 / (1.0 + (residual[current_valid] / sigma) ** 2)
            try:
                popt, _ = curve_fit(
                    func,
                    t[current_valid],
                    filled[current_valid],
                    p0=popt,
                    bounds=(lower, upper),
                    sigma=np.maximum(
                        1.0 / np.sqrt(np.maximum(weights[current_valid], 1e-06)),
                        0.001,
                    ),
                    maxfev=max_iterations,
                )
            except Exception:
                pass
        if outlier_method == '不处理':
            break
        new_valid = current_valid & ~bad
        if new_valid.sum() < min_valid:
            break
        if np.array_equal(new_valid, current_valid):
            break
        current_valid = new_valid
        p0 = popt.copy()
    if popt is None:
        return filled.astype(np.float32)
    try:
        result = func(t, *popt)
        result = np.asarray(result, dtype=np.float32)
        result[~np.isfinite(result)] = np.nan
        return result
    except Exception:
        return filled.astype(np.float32)


class PhenologyCalculator:

    def __init__(self, method, sos_threshold=0.3, eos_threshold=0.3):
        self.method = method
        self.sos_threshold = sos_threshold
        self.eos_threshold = eos_threshold

    def relative_threshold(self, y, sos_threshold, eos_threshold):
        if y is None:
            return (np.nan, np.nan)
        if not np.any(np.isfinite(y)):
            return (np.nan, np.nan)
        minimum = np.nanmin(y)
        maximum = np.nanmax(y)
        amplitude = maximum - minimum
        if amplitude <= 0:
            return (np.nan, np.nan)
        peak = int(np.nanargmax(y))
        sos_value = minimum + amplitude * sos_threshold
        eos_value = minimum + amplitude * eos_threshold
        sos = np.nan
        for i in range(0, peak + 1):
            if y[i] >= sos_value:
                sos = i
                break
        eos = np.nan
        for i in range(peak, len(y)):
            if y[i] <= eos_value:
                eos = i
                break
        return (sos, eos)

    def dynamic_threshold(self, y, sos_threshold, eos_threshold):
        if y is None:
            return (np.nan, np.nan)
        valid = y[np.isfinite(y)]
        if len(valid) < 3:
            return (np.nan, np.nan)
        p5 = np.percentile(valid, 5)
        p95 = np.percentile(valid, 95)
        amplitude = p95 - p5
        if amplitude <= 0:
            return (np.nan, np.nan)
        sos_value = p5 + amplitude * sos_threshold
        eos_value = p5 + amplitude * eos_threshold
        peak = int(np.nanargmax(y))
        sos = np.nan
        for i in range(peak + 1):
            if y[i] >= sos_value:
                sos = i
                break
        eos = np.nan
        for i in range(peak, len(y)):
            if y[i] <= eos_value:
                eos = i
                break
        return (sos, eos)

    def polynomial(self, y):
        filled = fill_nan_series(y)
        if filled is None:
            return (np.nan, np.nan)
        n = len(filled)
        x = np.arange(n)
        try:
            degree = min(3, n - 1)
            coef = np.polyfit(x, filled, degree)
            curve = np.polyval(coef, x)
            peak = int(np.argmax(curve))
            diff = np.gradient(curve)
            sos = np.nan
            for i in range(peak):
                if diff[i] > 0:
                    sos = i
            eos = np.nan
            for i in range(peak, n):
                if diff[i] < 0:
                    eos = i
                    break
            return (sos, eos)
        except Exception:
            return (np.nan, np.nan)

    def single(self, y):
        if self.method == '相对阈值':
            return self.relative_threshold(y, self.sos_threshold, self.eos_threshold)
        elif self.method == '动态阈值':
            return self.dynamic_threshold(y, self.sos_threshold, self.eos_threshold)
        elif self.method == '多项式拟合':
            return self.polynomial(y)
        return (np.nan, np.nan)


def period_to_doy(period_index):
    return min(365, max(1, 1 + 16 * int(period_index)))


def write_result(output_path, data, reference_profile, valid_mask=None):
    profile = reference_profile.copy()
    profile.update(dtype='float32', count=1, nodata=NODATA, compress='lzw')
    arr = np.asarray(data, dtype=np.float32).copy()
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != arr.shape:
            raise ValueError('结果数组与研究区掩膜尺寸不一致：{} != {}'.format(arr.shape, valid_mask.shape))
        arr[~valid_mask] = NODATA
    arr[~np.isfinite(arr)] = NODATA
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(arr, 1)


class ProgressDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('物候提取运行进度')
        self.setFixedSize(520, 330)
        layout = QtWidgets.QVBoxLayout(self)
        self.title_label = QtWidgets.QLabel('正在进行物候提取...')
        self.title_label.setStyleSheet('font-size:16px;font-weight:bold;')
        layout.addWidget(self.title_label)
        self.status_label = QtWidgets.QLabel('准备运行...')
        layout.addWidget(self.status_label)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        self.cancel_button = QtWidgets.QPushButton('停止运行')
        layout.addWidget(self.cancel_button)

    def set_status(self, text):
        self.status_label.setText(text)

    def set_progress(self, value):
        self.progress.setValue(int(value))

    def append_log(self, text):
        self.log_box.append(str(text))


class PhenologyWorker(QtCore.QThread):
    progress_signal = QtCore.Signal(int)
    status_signal = QtCore.Signal(str)
    log_signal = QtCore.Signal(str)
    finished_signal = QtCore.Signal(str)
    error_signal = QtCore.Signal(str)

    def __init__(
        self,
        input_folder,
        output_folder,
        start_year,
        end_year,
        index_type,
        filter_type,
        method,
        sos_threshold,
        eos_threshold,
        season_mode,
        region_mask_path,
        second_mask_path=None,
        sg_window=7,
        sg_polyorder=2,
        sg_missing_method='线性插值',
        hants_harmonics=2,
        hants_iterations=500,
        hants_outlier_threshold=0.05,
        hants_outlier_direction='双向异常值',
        hants_min_valid=8,
        hants_trend='不使用',
        hants_weight='等权',
        logistic_model='标准双逻辑斯蒂函数',
        logistic_initial='自动估计',
        logistic_iterations=500,
        logistic_tolerance=1e-05,
        logistic_outlier='残差法',
        logistic_min_valid=8,
        logistic_weight='等权',
        parent=None,
    ):
        super().__init__(parent)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.start_year = int(start_year)
        self.end_year = int(end_year)
        self.index_type = index_type
        self.filter_type = filter_type
        self.method = method
        self.sos_threshold = float(sos_threshold)
        self.eos_threshold = float(eos_threshold)
        self.season_mode = season_mode
        self.region_mask_path = region_mask_path
        self.second_mask_path = second_mask_path
        self.sg_window = int(sg_window)
        self.sg_polyorder = int(sg_polyorder)
        self.sg_missing_method = sg_missing_method
        self.hants_harmonics = int(hants_harmonics)
        self.hants_iterations = int(hants_iterations)
        self.hants_outlier_threshold = float(hants_outlier_threshold)
        self.hants_outlier_direction = hants_outlier_direction
        self.hants_min_valid = int(hants_min_valid)
        self.hants_trend = hants_trend
        self.hants_weight = hants_weight
        self.logistic_model = logistic_model
        self.logistic_initial = logistic_initial
        self.logistic_iterations = int(logistic_iterations)
        self.logistic_tolerance = float(logistic_tolerance)
        self.logistic_outlier = logistic_outlier
        self.logistic_min_valid = int(logistic_min_valid)
        self.logistic_weight = logistic_weight
        self._stop = False

    def stop(self):
        self._stop = True

    def _read_mask(self, path, reference_raster):
        with rasterio.open(reference_raster) as reference:
            shape = (reference.height, reference.width)
            transform = reference.transform
            crs = reference.crs
        with rasterio.open(path) as mask_ds:
            if (mask_ds.height, mask_ds.width) != shape:
                raise ValueError('掩膜与EVI/NDVI栅格尺寸不一致：\n{}'.format(path))
            if mask_ds.transform != transform:
                raise ValueError('掩膜与EVI/NDVI空间变换参数不一致：\n{}'.format(path))
            if mask_ds.crs != crs:
                raise ValueError('掩膜与EVI/NDVI坐标系不一致：\n{}'.format(path))
            mask_data = mask_ds.read(1)
            inside = np.isfinite(mask_data)
            if mask_ds.nodata is not None:
                inside &= mask_data != mask_ds.nodata
            inside &= mask_data != 0
            return inside

    def _filter_stack(self, stack, region_mask):
        filtered = np.full_like(stack, np.nan)
        usable = region_mask & np.any(np.isfinite(stack), axis=0)
        row_ids, col_ids = np.where(usable)
        pixel_count = len(row_ids)
        for pos, (row, col) in enumerate(zip(row_ids, col_ids)):
            if self._stop:
                return (None, row_ids, col_ids)
            series = stack[:, row, col]
            if self.filter_type == '不进行滤波':
                smoothed = fill_nan_series(series)
            elif self.filter_type == 'S-G滤波':
                smoothed = sg_filter_series(
                    series,
                    self.sg_window,
                    self.sg_polyorder,
                    self.sg_missing_method,
                )
            elif self.filter_type == 'HANTS滤波':
                smoothed = hants_series(
                    series,
                    harmonics=self.hants_harmonics,
                    max_iterations=self.hants_iterations,
                    outlier_threshold=self.hants_outlier_threshold,
                    outlier_direction=self.hants_outlier_direction,
                    min_valid=self.hants_min_valid,
                    trend=self.hants_trend,
                    weight_mode=self.hants_weight,
                )
            elif self.filter_type == '双逻辑斯蒂函数拟合':
                smoothed = double_logistic_fit(
                    series,
                    model_type=self.logistic_model,
                    initial_mode=self.logistic_initial,
                    max_iterations=self.logistic_iterations,
                    tolerance=self.logistic_tolerance,
                    outlier_method=self.logistic_outlier,
                    min_valid=self.logistic_min_valid,
                    weight_mode=self.logistic_weight,
                )
            else:
                smoothed = fill_nan_series(series)
            if smoothed is not None:
                filtered[:, row, col] = smoothed
            progress = 20 + (pos + 1) / max(pixel_count, 1) * 25
            self.progress_signal.emit(int(progress))
        return (filtered, row_ids, col_ids)

    def _extract(
        self,
        filtered,
        target_mask,
        height,
        width,
        label,
        progress_start,
        progress_end,
    ):
        calculator = PhenologyCalculator(self.method, self.sos_threshold, self.eos_threshold)
        sos = np.full((height, width), NODATA, dtype=np.float32)
        eos = np.full((height, width), NODATA, dtype=np.float32)
        usable = target_mask & np.any(np.isfinite(filtered), axis=0)
        row_ids, col_ids = np.where(usable)
        pixel_count = len(row_ids)
        self.status_signal.emit('正在提取{}SOS/EOS...'.format(label))
        self.log_signal.emit('{}有效计算像元：{}'.format(label, pixel_count))
        span = progress_end - progress_start
        for pos, (row, col) in enumerate(zip(row_ids, col_ids)):
            if self._stop:
                return (None, None)
            series = filtered[:, row, col]
            if not np.any(np.isfinite(series)):
                continue
            start_index, end_index = calculator.single(series)
            if np.isfinite(start_index):
                sos[row, col] = period_to_doy(start_index)
            if np.isfinite(end_index):
                eos[row, col] = period_to_doy(end_index)
            progress = progress_start + (pos + 1) / max(pixel_count, 1) * span
            self.progress_signal.emit(int(progress))
        return (sos, eos)

    def process_year(self, year, region_mask, second_mask=None):
        if self._stop:
            return False
        self.status_signal.emit('正在处理{}年数据...'.format(year))
        self.log_signal.emit('开始处理{}年'.format(year))
        year_info = find_year_files(self.input_folder, year, self.index_type)
        if not year_info['ok']:
            raise ValueError(year_info['message'])
        year_files = year_info['files']
        reference_layer, profile, _, _ = read_raster_scaled(year_files[0])
        height, width = reference_layer.shape
        if region_mask.shape != (height, width):
            raise ValueError('研究区掩膜尺寸与{}年EVI/NDVI栅格不一致。'.format(year))
        stack = np.full((PERIOD_COUNT, height, width), np.nan, dtype=np.float32)
        for i, raster_path in enumerate(year_files):
            if self._stop:
                return False
            layer, _, _, _ = read_raster_scaled(raster_path)
            if layer.shape != (height, width):
                raise ValueError('{}年第{}期栅格尺寸不一致。'.format(year, i + 1))
            stack[i] = layer
            self.progress_signal.emit(int((i + 1) / PERIOD_COUNT * 20))
        stack[~np.broadcast_to(region_mask, stack.shape)] = np.nan
        self.status_signal.emit('{}年：正在进行数据预处理（{}）'.format(year, self.filter_type))
        self.log_signal.emit('滤波方式：{}'.format(self.filter_type))
        if self.filter_type == 'S-G滤波':
            self.log_signal.emit('S-G参数：窗口{}，阶数{}，{}'.format(self.sg_window, self.sg_polyorder, self.sg_missing_method))
        elif self.filter_type == 'HANTS滤波':
            self.log_signal.emit(
                'HANTS参数：谐波{}，迭代{}，阈值{}，{}，最小有效{}，{}，{}'.format(
                    self.hants_harmonics,
                    self.hants_iterations,
                    self.hants_outlier_threshold,
                    self.hants_outlier_direction,
                    self.hants_min_valid,
                    self.hants_trend,
                    self.hants_weight,
                )
            )
        elif self.filter_type == '双逻辑斯蒂函数拟合':
            self.log_signal.emit(
                '双逻辑斯蒂参数：{}，{}，迭代{}，容差{}，{}，最小有效{}，{}'.format(
                    self.logistic_model,
                    self.logistic_initial,
                    self.logistic_iterations,
                    self.logistic_tolerance,
                    self.logistic_outlier,
                    self.logistic_min_valid,
                    self.logistic_weight,
                )
            )
        filtered, _, _ = self._filter_stack(stack, region_mask)
        if filtered is None:
            return False
        first_sos, first_eos = self._extract(filtered, region_mask, height, width, '第一生长季', 45, 80)
        if first_sos is None:
            return False
        first_sos[~region_mask] = NODATA
        first_eos[~region_mask] = NODATA
        second_sos = second_eos = None
        second_target_mask = None
        if self.season_mode == '双季':
            if second_mask is None:
                raise ValueError('双季模式未提供第二生长季掩膜。')
            # second_target_mask = second_mask  # 旧版未限制到当前研究区
            second_target_mask = region_mask & second_mask
            self.log_signal.emit('第二生长季有效像元（研究区与双季掩膜交集）：{}'.format(int(np.count_nonzero(second_target_mask))))
            second_sos, second_eos = self._extract(filtered, second_target_mask, height, width, '第二生长季', 80, 95)
            if second_sos is None:
                return False
            second_sos[~second_target_mask] = NODATA
            second_eos[~second_target_mask] = NODATA
        self.status_signal.emit('{}年：正在保存SOS/EOS结果...'.format(year))
        os.makedirs(self.output_folder, exist_ok=True)
        write_result(os.path.join(self.output_folder, 'SOS1_{}.tif'.format(year)), first_sos, profile, region_mask)
        write_result(os.path.join(self.output_folder, 'EOS1_{}.tif'.format(year)), first_eos, profile, region_mask)
        if self.season_mode == '双季':
            write_result(
                os.path.join(self.output_folder, 'SOS2_{}.tif'.format(year)),
                second_sos,
                profile,
                second_target_mask,
            )
            write_result(
                os.path.join(self.output_folder, 'EOS2_{}.tif'.format(year)),
                second_eos,
                profile,
                second_target_mask,
            )
        self.progress_signal.emit(100)
        self.log_signal.emit('{}年处理完成。'.format(year))
        return True

    def run(self):
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            years = list(range(self.start_year, self.end_year + 1))
            self.status_signal.emit('正在检查年度数据...')
            self.log_signal.emit('开始检查{}—{}年数据。'.format(self.start_year, self.end_year))
            year_files = {}
            for year in years:
                result = find_year_files(self.input_folder, year, self.index_type)
                if not result['ok']:
                    raise ValueError(result['message'])
                year_files[year] = result['files']
                self.log_signal.emit('{}年：23期数据检查通过。'.format(year))
            if self._stop:
                return
            reference = year_files[self.start_year][0]
            region_mask = self._read_mask(self.region_mask_path, reference)
            region_pixel_count = int(np.count_nonzero(region_mask))
            if region_pixel_count == 0:
                raise ValueError('研究区掩膜没有有效像元，请检查SHP边界与输入栅格是否重叠。')
            self.log_signal.emit('SHP研究区有效像元：{}；边界外像元将写为NoData。'.format(region_pixel_count))
            second_mask = None
            if self.season_mode == '双季':
                second_mask = self._read_mask(self.second_mask_path, reference)
                second_mask = region_mask & second_mask
                if not np.any(second_mask):
                    raise ValueError('第二生长季掩膜与当前选择的研究区没有重叠有效像元。')
            all_files = []
            for y in years:
                all_files.extend(year_files[y])
            self.status_signal.emit('正在检查栅格空间一致性...')
            check_raster_consistency(all_files)
            for yi, year in enumerate(years):
                if self._stop:
                    return
                ok = self.process_year(year, region_mask, second_mask)
                if not ok:
                    if self._stop:
                        self.log_signal.emit('用户停止运行。')
                        return
                    raise RuntimeError('{}年处理未完成。'.format(year))
                self.progress_signal.emit(int((yi + 1) / len(years) * 100))
            self.status_signal.emit('全部处理完成！')
            self.log_signal.emit('================================')
            self.log_signal.emit('全部年份物候提取完成。')
            self.finished_signal.emit(self.output_folder)
        except Exception as e:
            self.error_signal.emit(str(e) + '\n\n' + traceback.format_exc())


class PhenologySoftware(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('多源遥感植被物候提取与时序分析系统  V1.0')
        self.resize(1150, 760)
        self.region_manager = RegionManager()
        self.worker = None
        self.progress_dialog = None
        self.region_mask_path = None
        self.second_mask_path = None
        self.current_result_files = []
        self.build_ui()

    def build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        # 移除功能区上方的大标题，将空间留给五个子界面。
        # 完整系统名称与版本号保留在窗口标题栏。
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)
        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs, 1)
        self.create_data_tab()
        self.create_preprocess_tab()
        self.create_phenology_tab()
        self.create_analysis_tab()
        self.create_view_tab()
        # 五个子界面统一留白与控件间距，随窗口大小调整。
        for index in range(self.tabs.count()):
            page_layout = self.tabs.widget(index).layout()
            if page_layout is not None:
                page_layout.setContentsMargins(12, 12, 12, 12)
                page_layout.setSpacing(10)

    def create_data_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        form = QtWidgets.QFormLayout()
        self.index_combo = QtWidgets.QComboBox()
        self.index_combo.addItems(['EVI', 'NDVI'])
        form.addRow('数据类型：', self.index_combo)
        il = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QLineEdit()
        b = QtWidgets.QPushButton('选择')
        b.clicked.connect(self.select_input_folder)
        il.addWidget(self.input_edit)
        il.addWidget(b)
        form.addRow('输入数据目录：', il)
        ol = QtWidgets.QHBoxLayout()
        self.output_edit = QtWidgets.QLineEdit()
        b = QtWidgets.QPushButton('选择')
        b.clicked.connect(self.select_output_folder)
        ol.addWidget(self.output_edit)
        ol.addWidget(b)
        form.addRow('输出目录：', ol)
        yl = QtWidgets.QHBoxLayout()
        self.start_year = QtWidgets.QSpinBox()
        self.start_year.setRange(MIN_YEAR, MAX_YEAR)
        self.start_year.setValue(2000)
        self.end_year = QtWidgets.QSpinBox()
        self.end_year.setRange(MIN_YEAR, MAX_YEAR)
        self.end_year.setValue(2003)
        yl.addWidget(QtWidgets.QLabel('开始年份'))
        yl.addWidget(self.start_year)
        yl.addWidget(QtWidgets.QLabel('结束年份'))
        yl.addWidget(self.end_year)
        form.addRow('年份范围：', yl)
        layout.addLayout(form)
        self.check_data_button = QtWidgets.QPushButton('检查年度数据')
        self.check_data_button.clicked.connect(self.check_year_data)
        layout.addWidget(self.check_data_button)
        self.data_log = QtWidgets.QTextEdit()
        self.data_log.setReadOnly(True)
        layout.addWidget(self.data_log)
        layout.addStretch()
        self.tabs.addTab(page, '数据管理')

    def create_preprocess_tab(self):
        page = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        group = QtWidgets.QGroupBox('数据预处理方式')
        gl = QtWidgets.QVBoxLayout(group)
        self.filter_group = QtWidgets.QButtonGroup(self)
        self.filter_none = QtWidgets.QRadioButton('不进行滤波')
        self.filter_sg = QtWidgets.QRadioButton('S-G滤波')
        self.filter_hants = QtWidgets.QRadioButton('HANTS滤波')
        self.filter_logistic = QtWidgets.QRadioButton('双逻辑斯蒂函数拟合')
        self.filter_none.setChecked(True)
        for x in [self.filter_none, self.filter_sg, self.filter_hants, self.filter_logistic]:
            self.filter_group.addButton(x)
            gl.addWidget(x)
        layout.addWidget(group)
        self.filter_description = QtWidgets.QLabel()
        self.filter_description.setWordWrap(True)
        self.filter_description.setStyleSheet('padding:8px;border:1px solid #cccccc;background:#f5f5f5;')
        layout.addWidget(self.filter_description)
        self.sg_group = QtWidgets.QGroupBox('S-G滤波参数')
        sl = QtWidgets.QGridLayout(self.sg_group)
        self.sg_window_combo = QtWidgets.QComboBox()
        self.sg_window_combo.addItems(['5', '7', '9', '11', '13', '15', '17', '19', '21'])
        self.sg_window_combo.setCurrentText('7')
        sl.addWidget(QtWidgets.QLabel('窗口大小：'), 0, 0)
        sl.addWidget(self.sg_window_combo, 0, 1)
        self.sg_order_combo = QtWidgets.QComboBox()
        self.sg_order_combo.addItems(['1', '2', '3', '4'])
        self.sg_order_combo.setCurrentText('2')
        sl.addWidget(QtWidgets.QLabel('多项式阶数：'), 1, 0)
        sl.addWidget(self.sg_order_combo, 1, 1)
        self.sg_missing_combo = QtWidgets.QComboBox()
        self.sg_missing_combo.addItems(['线性插值'])
        sl.addWidget(QtWidgets.QLabel('缺失值处理：'), 2, 0)
        sl.addWidget(self.sg_missing_combo, 2, 1)
        sl.addWidget(QtWidgets.QLabel('说明：23期MOD13A2时间序列建议优先使用窗口7、多项式2。窗口越大，平滑程度通常越高。'), 3, 0, 1, 2)
        layout.addWidget(self.sg_group)
        self.hants_group = QtWidgets.QGroupBox('HANTS滤波参数')
        hl = QtWidgets.QGridLayout(self.hants_group)

        def combo(items, default):
            c = QtWidgets.QComboBox()
            c.addItems(items)
            c.setCurrentText(default)
            return c
        self.hants_harmonics_combo = combo(['1', '2', '3', '4', '5'], '2')
        self.hants_iterations_combo = combo(['100', '200', '500', '1000'], '500')
        self.hants_threshold_combo = combo(['0.01', '0.02', '0.05', '0.10', '0.15', '0.20'], '0.05')
        self.hants_direction_combo = combo(['仅高异常值', '仅低异常值', '双向异常值'], '双向异常值')
        self.hants_min_valid_combo = combo(['5', '6', '8', '10', '12', '15'], '8')
        self.hants_trend_combo = combo(['不使用', '一阶趋势'], '不使用')
        self.hants_weight_combo = combo(['等权', '鲁棒权重'], '等权')
        vals = [
            ('谐波阶数：', self.hants_harmonics_combo),
            ('最大迭代次数：', self.hants_iterations_combo),
            ('异常值阈值：', self.hants_threshold_combo),
            ('异常值方向：', self.hants_direction_combo),
            ('最小有效观测数：', self.hants_min_valid_combo),
            ('趋势项：', self.hants_trend_combo),
            ('拟合权重：', self.hants_weight_combo),
        ]
        for i, (lab, w) in enumerate(vals):
            hl.addWidget(QtWidgets.QLabel(lab), i, 0)
            hl.addWidget(w, i, 1)
        hl.addWidget(QtWidgets.QLabel('说明：对于23期MOD13A2数据，建议谐波阶数2、迭代500次、异常值阈值0.05。'), 7, 0, 1, 2)
        layout.addWidget(self.hants_group)
        self.logistic_group = QtWidgets.QGroupBox('双逻辑斯蒂函数拟合参数')
        ll = QtWidgets.QGridLayout(self.logistic_group)
        self.logistic_model_combo = combo(['标准双逻辑斯蒂函数', '非对称双逻辑斯蒂函数'], '标准双逻辑斯蒂函数')
        self.logistic_initial_combo = combo(['自动估计', '固定范围'], '自动估计')
        self.logistic_iterations_combo = combo(['100', '200', '500', '1000'], '500')
        self.logistic_tolerance_combo = combo(['1e-3', '1e-4', '1e-5', '1e-6'], '1e-5')
        self.logistic_outlier_combo = combo(['不处理', '残差法', 'IQR法'], '残差法')
        self.logistic_min_valid_combo = combo(['5', '6', '8', '10', '12', '15'], '8')
        self.logistic_weight_combo = combo(['等权', '鲁棒权重'], '等权')
        vals = [
            ('拟合模型：', self.logistic_model_combo),
            ('初始参数：', self.logistic_initial_combo),
            ('最大迭代次数：', self.logistic_iterations_combo),
            ('收敛容差：', self.logistic_tolerance_combo),
            ('异常值处理：', self.logistic_outlier_combo),
            ('最小有效观测数：', self.logistic_min_valid_combo),
            ('拟合权重：', self.logistic_weight_combo),
        ]
        for i, (lab, w) in enumerate(vals):
            ll.addWidget(QtWidgets.QLabel(lab), i, 0)
            ll.addWidget(w, i, 1)
        ll.addWidget(QtWidgets.QLabel('说明：标准双逻辑斯蒂适合常规季节曲线；非对称模型允许春季和秋季变化速率不同。'), 7, 0, 1, 2)
        layout.addWidget(self.logistic_group)
        layout.addWidget(QtWidgets.QLabel('提示：四种预处理方式始终显示。选择其中一种滤波方法后，仅对应参数用于物候提取。'))
        layout.addStretch()
        scroll.setWidget(content)
        pl = QtWidgets.QVBoxLayout(page)
        pl.addWidget(scroll)
        self.tabs.addTab(page, '数据预处理')
        for x in [self.filter_none, self.filter_sg, self.filter_hants, self.filter_logistic]:
            x.toggled.connect(self.update_filter_parameter_panels)
        self.update_filter_parameter_panels()

    def update_filter_parameter_panels(self, checked=None):
        sg = self.filter_sg.isChecked()
        ha = self.filter_hants.isChecked()
        lo = self.filter_logistic.isChecked()
        self.sg_group.setVisible(sg)
        self.hants_group.setVisible(ha)
        self.logistic_group.setVisible(lo)
        if sg:
            t = '当前选择：S-G滤波。下方显示S-G滤波参数，可设置窗口大小、多项式阶数和缺失值处理方式。'
        elif ha:
            t = '当前选择：HANTS滤波。下方显示谐波阶数、异常值处理、迭代次数、趋势项和拟合权重等参数。'
        elif lo:
            t = '当前选择：双逻辑斯蒂函数拟合。下方显示拟合模型、初始参数、迭代次数、收敛容差、异常值处理和权重等参数。'
        else:
            t = '当前选择：不进行滤波。原始时间序列仅进行缺失值线性插值，随后直接进行物候期提取。'
        self.filter_description.setText(t)

    def create_phenology_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        rg = QtWidgets.QGroupBox('研究区域设置')
        rl = QtWidgets.QGridLayout(rg)
        self.region_type_combo = QtWidgets.QComboBox()
        self.region_type_combo.addItems(['SHP矢量区域', 'TIF栅格掩膜'])
        self.region_type_combo.currentTextChanged.connect(self.region_type_changed)
        rl.addWidget(QtWidgets.QLabel('研究区类型：'), 0, 0)
        rl.addWidget(self.region_type_combo, 0, 1)
        self.shp_edit = QtWidgets.QLineEdit()
        self.region_file_button = QtWidgets.QPushButton('选择文件')
        self.region_file_button.clicked.connect(self.select_region_file)
        self.load_region_button = QtWidgets.QPushButton('读取属性字段')
        self.load_region_button.clicked.connect(self.load_shp_fields)
        rl.addWidget(QtWidgets.QLabel('研究区文件：'), 1, 0)
        rl.addWidget(self.shp_edit, 1, 1)
        rl.addWidget(self.region_file_button, 1, 2)
        rl.addWidget(self.load_region_button, 1, 3)
        self.region_all_radio = QtWidgets.QRadioButton('SHP全部面（按边界提取）')
        self.region_select_radio = QtWidgets.QRadioButton('选择属性区域')
        self.region_all_radio.setChecked(True)
        bg = QtWidgets.QButtonGroup(self)
        bg.addButton(self.region_all_radio)
        bg.addButton(self.region_select_radio)
        rl.addWidget(self.region_all_radio, 2, 0)
        rl.addWidget(self.region_select_radio, 2, 1)
        self.field_combo = QtWidgets.QComboBox()
        self.field_combo.currentTextChanged.connect(self.field_changed)
        rl.addWidget(QtWidgets.QLabel('属性字段：'), 3, 0)
        rl.addWidget(self.field_combo, 3, 1)
        self.region_combo = QtWidgets.QComboBox()
        rl.addWidget(QtWidgets.QLabel('选择区域：'), 4, 0)
        rl.addWidget(self.region_combo, 4, 1)
        self.mask_button = QtWidgets.QPushButton('生成研究区栅格掩膜')
        self.mask_button.clicked.connect(self.generate_mask)
        rl.addWidget(self.mask_button, 5, 0, 1, 2)
        self.mask_label = QtWidgets.QLabel('研究区掩膜：尚未生成')
        self.mask_label.setWordWrap(True)
        rl.addWidget(self.mask_label, 6, 0, 1, 4)
        self.region_rule_label = QtWidgets.QLabel('空间规则：只计算SHP面内像元；输出保持原始栅格网格，SHP边界外统一写为NoData（-9999）。')
        self.region_rule_label.setWordWrap(True)
        rl.addWidget(self.region_rule_label, 7, 0, 1, 4)
        layout.addWidget(rg)
        self.shp_edit.textChanged.connect(self.invalidate_region_mask)
        self.region_all_radio.toggled.connect(self.invalidate_region_mask)
        self.region_select_radio.toggled.connect(self.invalidate_region_mask)
        self.region_combo.currentTextChanged.connect(self.invalidate_region_mask)
        self.input_edit.textChanged.connect(self.invalidate_region_mask)
        self.index_combo.currentTextChanged.connect(self.invalidate_region_mask)
        self.start_year.valueChanged.connect(self.invalidate_region_mask)
        mg = QtWidgets.QGroupBox('物候提取参数')
        ml = QtWidgets.QGridLayout(mg)
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(['相对阈值', '动态阈值', '多项式拟合'])
        self.method_combo.currentTextChanged.connect(self.method_changed)
        ml.addWidget(QtWidgets.QLabel('物候方法：'), 0, 0)
        ml.addWidget(self.method_combo, 0, 1)
        self.relative_group = QtWidgets.QGroupBox('相对阈值')
        x = QtWidgets.QGridLayout(self.relative_group)
        self.relative_sos = QtWidgets.QDoubleSpinBox()
        self.relative_eos = QtWidgets.QDoubleSpinBox()
        for z in [self.relative_sos, self.relative_eos]:
            z.setRange(0.05, 0.95)
            z.setSingleStep(0.05)
            z.setDecimals(2)
            z.setValue(0.3)
        self.relative_sos.valueChanged.connect(self.sync_relative_sos)
        self.relative_eos.valueChanged.connect(self.sync_relative_eos)
        x.addWidget(QtWidgets.QLabel('SOS阈值：'), 0, 0)
        x.addWidget(self.relative_sos, 0, 1)
        x.addWidget(QtWidgets.QLabel('EOS阈值：'), 1, 0)
        x.addWidget(self.relative_eos, 1, 1)
        ml.addWidget(self.relative_group, 1, 0, 1, 2)
        self.dynamic_group = QtWidgets.QGroupBox('动态阈值')
        x = QtWidgets.QGridLayout(self.dynamic_group)
        self.dynamic_sos = QtWidgets.QComboBox()
        self.dynamic_eos = QtWidgets.QComboBox()
        vals = ['0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50']
        self.dynamic_sos.addItems(vals)
        self.dynamic_eos.addItems(vals)
        self.dynamic_sos.setCurrentText('0.30')
        self.dynamic_eos.setCurrentText('0.30')
        x.addWidget(QtWidgets.QLabel('SOS阈值：'), 0, 0)
        x.addWidget(self.dynamic_sos, 0, 1)
        x.addWidget(QtWidgets.QLabel('EOS阈值：'), 1, 0)
        x.addWidget(self.dynamic_eos, 1, 1)
        ml.addWidget(self.dynamic_group, 2, 0, 1, 2)
        ml.addWidget(QtWidgets.QLabel('生长季数量：'), 3, 0)
        self.season_combo = QtWidgets.QComboBox()
        self.season_combo.addItems(['单季', '双季'])
        self.season_combo.currentTextChanged.connect(self.season_mode_changed)
        ml.addWidget(self.season_combo, 3, 1)
        layout.addWidget(mg)
        self.second_group = QtWidgets.QGroupBox('第二生长季掩膜（双季模式）')
        sl = QtWidgets.QGridLayout(self.second_group)
        self.second_mask_edit = QtWidgets.QLineEdit()
        self.second_mask_button = QtWidgets.QPushButton('选择TIF/SHP')
        self.second_mask_button.clicked.connect(self.select_second_mask)
        self.second_generate_button = QtWidgets.QPushButton('生成第二生长季掩膜')
        self.second_generate_button.clicked.connect(self.generate_second_mask)
        sl.addWidget(QtWidgets.QLabel('第二生长季文件：'), 0, 0)
        sl.addWidget(self.second_mask_edit, 0, 1)
        sl.addWidget(self.second_mask_button, 0, 2)
        self.second_generate_button.setToolTip('第二季有效区将自动限制在当前选择的研究区内')
        sl.addWidget(self.second_generate_button, 1, 0, 1, 3)
        self.second_mask_label = QtWidgets.QLabel('第二生长季掩膜：尚未生成。最终有效区为研究区与该掩膜的交集。')
        self.second_mask_label.setWordWrap(True)
        sl.addWidget(self.second_mask_label, 2, 0, 1, 3)
        layout.addWidget(self.second_group)
        run = QtWidgets.QGroupBox('物候提取运行')
        q = QtWidgets.QVBoxLayout(run)
        self.run_button = QtWidgets.QPushButton('开始运行')
        self.run_button.setMinimumHeight(50)
        self.run_button.setStyleSheet('font-size:18px;font-weight:bold;')
        self.run_button.clicked.connect(self.start_phenology)
        q.addWidget(self.run_button)
        self.run_info = QtWidgets.QLabel('请完成以上参数选择后点击“开始运行”。')
        self.run_info.setWordWrap(True)
        q.addWidget(self.run_info)
        layout.addWidget(run)
        layout.addStretch()
        self.tabs.addTab(page, '物候提取')
        self.method_changed(self.method_combo.currentText())
        self.season_mode_changed(self.season_combo.currentText())

    def create_analysis_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        top = QtWidgets.QGroupBox('本次计算的SOS/EOS物候影像')
        tl = QtWidgets.QVBoxLayout(top)
        row = QtWidgets.QHBoxLayout()
        self.result_combo = QtWidgets.QComboBox()
        self.result_combo.currentTextChanged.connect(self.display_result_image)
        row.addWidget(QtWidgets.QLabel('选择影像：'))
        row.addWidget(self.result_combo)
        self.analysis_refresh_button = QtWidgets.QPushButton('刷新结果')
        self.analysis_refresh_button.clicked.connect(self.refresh_analysis_results)
        row.addWidget(self.analysis_refresh_button)
        tl.addLayout(row)
        self.result_image_label = QtWidgets.QLabel('完成计算后将在此显示SOS/EOS影像')
        self.result_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_image_label.setMinimumHeight(300)
        self.result_image_label.setStyleSheet('border:1px solid #cccccc;background:#f5f5f5;')
        tl.addWidget(self.result_image_label)
        self.result_info = QtWidgets.QLabel('')
        self.result_info.setWordWrap(True)
        tl.addWidget(self.result_info)
        layout.addWidget(top)
        bottom = QtWidgets.QGroupBox('物候影像统计值')
        bl = QtWidgets.QVBoxLayout(bottom)
        self.analysis_table = QtWidgets.QTableWidget()
        self.analysis_table.setColumnCount(9)
        self.analysis_table.setHorizontalHeaderLabels(
            ['影像', '有效像元', '最小值', '最大值', '平均值', '中位数', '方差', '标准差', 'NoData像元']
        )
        bl.addWidget(self.analysis_table)
        layout.addWidget(bottom)
        self.tabs.addTab(page, '结果分析')

    def create_view_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        g = QtWidgets.QGroupBox('外部物候影像分析')
        gl = QtWidgets.QGridLayout(g)
        self.external_edit = QtWidgets.QLineEdit()
        b = QtWidgets.QPushButton('选择TIF')
        b.clicked.connect(self.select_external_tif)
        load = QtWidgets.QPushButton('加载影像')
        load.clicked.connect(self.load_external_tif)
        gl.addWidget(QtWidgets.QLabel('物候影像：'), 0, 0)
        gl.addWidget(self.external_edit, 0, 1)
        gl.addWidget(b, 0, 2)
        gl.addWidget(load, 0, 3)
        self.external_image_label = QtWidgets.QLabel('请选择任意外部物候TIF')
        self.external_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.external_image_label.setMinimumHeight(300)
        self.external_image_label.setStyleSheet('border:1px solid #cccccc;background:#f5f5f5;')
        gl.addWidget(self.external_image_label, 1, 0, 1, 4)
        self.external_info = QtWidgets.QLabel('')
        self.external_info.setWordWrap(True)
        gl.addWidget(self.external_info, 2, 0, 1, 4)
        layout.addWidget(g)
        eg = QtWidgets.QGroupBox('外部影像统计值')
        el = QtWidgets.QVBoxLayout(eg)
        self.external_table = QtWidgets.QTableWidget()
        self.external_table.setColumnCount(9)
        self.external_table.setHorizontalHeaderLabels(
            ['影像', '有效像元', '最小值', '最大值', '平均值', '中位数', '方差', '标准差', 'NoData像元']
        )
        el.addWidget(self.external_table)
        layout.addWidget(eg)
        layout.addStretch()
        self.tabs.addTab(page, '外部影像分析')

    def select_input_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, '选择输入数据目录')
        if folder:
            self.input_edit.setText(folder)

    def select_output_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, '选择输出目录')
        if folder:
            self.output_edit.setText(folder)

    def check_year_data(self):
        input_folder = self.input_edit.text().strip()
        if not input_folder:
            QtWidgets.QMessageBox.warning(self, '提示', '请选择输入数据目录。')
            return
        start, end = (self.start_year.value(), self.end_year.value())
        if start > end:
            QtWidgets.QMessageBox.warning(self, '提示', '开始年份不能大于结束年份。')
            return
        self.data_log.clear()
        ok = True
        idx = self.index_combo.currentText()
        for year in range(start, end + 1):
            r = find_year_files(input_folder, year, idx)
            self.data_log.append(r['message'])
            if not r['ok']:
                ok = False
                if r['missing']:
                    self.data_log.append('缺失期数：{}'.format(','.join(map(str, r['missing']))))
                if r['duplicates']:
                    self.data_log.append('重复期数：{}'.format(','.join(map(str, r['duplicates'].keys()))))
        if ok:
            self.data_log.append('')
            self.data_log.append('所有年份数据检查通过。')
            QtWidgets.QMessageBox.information(self, '检查完成', '所有年份均为23期数据。')
        else:
            QtWidgets.QMessageBox.warning(self, '数据检查', '部分年份存在数据问题，请查看检查结果。')

    def region_type_changed(self, text):
        shp = text == 'SHP矢量区域'
        self.load_region_button.setVisible(shp)
        self.field_combo.setEnabled(shp)
        self.region_combo.setEnabled(shp)
        self.region_all_radio.setEnabled(shp)
        self.region_select_radio.setEnabled(shp)
        self.region_file_button.setText('选择SHP' if shp else '选择TIF')
        if not shp:
            self.field_combo.clear()
            self.region_combo.clear()
            self.region_all_radio.setChecked(True)
        self.invalidate_region_mask()

    def invalidate_region_mask(self, *args):
        self.region_mask_path = None
        if hasattr(self, 'mask_label'):
            self.mask_label.setText('研究区掩膜：参数已变化，运行前将按当前SHP/TIF重新生成。')

    def select_region_file(self):
        if self.region_type_combo.currentText() == 'SHP矢量区域':
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择研究区SHP', '', 'Shapefile (*.shp)')
            if path:
                self.shp_edit.setText(path)
                self.load_shp_fields()
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择研究区TIF', '', 'GeoTIFF (*.tif *.tiff)')
            if path:
                self.shp_edit.setText(path)
                self.region_mask_path = None
                self.mask_label.setText('研究区TIF已选择，运行前将重新生成掩膜：{}'.format(path))

    def select_shp(self):
        self.region_type_combo.setCurrentText('SHP矢量区域')
        self.select_region_file()

    def load_shp_fields(self):
        path = self.shp_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, '提示', '请先选择SHP文件。')
            return
        try:
            fields = self.region_manager.load_shp(path)
            self.field_combo.clear()
            self.field_combo.addItems(fields)
            self.invalidate_region_mask()
            if fields:
                self.field_changed(fields[0])
                QtWidgets.QMessageBox.information(self, '读取完成', 'SHP属性字段读取完成。')
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, '读取SHP失败', str(e))

    def field_changed(self, field):
        self.region_combo.clear()
        if field and self.region_manager.gdf is not None:
            self.region_combo.addItems(self.region_manager.get_field_values(field))
        self.invalidate_region_mask()

    def method_changed(self, text):
        self.relative_group.setVisible(text == '相对阈值')
        self.dynamic_group.setVisible(text == '动态阈值')

    def sync_relative_sos(self, value):
        if abs(self.relative_eos.value() - value) > 1e-09:
            self.relative_eos.blockSignals(True)
            self.relative_eos.setValue(value)
            self.relative_eos.blockSignals(False)

    def sync_relative_eos(self, value):
        if abs(self.relative_sos.value() - value) > 1e-09:
            self.relative_sos.blockSignals(True)
            self.relative_sos.setValue(value)
            self.relative_sos.blockSignals(False)

    def season_mode_changed(self, text):
        self.second_group.setVisible(text == '双季')

    def _reference_raster(self):
        r = find_year_files(self.input_edit.text().strip(), self.start_year.value(), self.index_combo.currentText())
        if not r['ok']:
            raise ValueError(r['message'])
        return r['files'][0]

    def _make_mask_from_file(self, file_path, output_path, allow_shp=True):
        ref = self._reference_raster()
        ext = Path(file_path).suffix.lower()
        if ext in ('.tif', '.tiff'):
            with rasterio.open(ref) as rr:
                shape = (rr.height, rr.width)
                transform = rr.transform
                crs = rr.crs
                profile = rr.profile.copy()
            with rasterio.open(file_path) as src:
                if (src.height, src.width) != shape:
                    raise ValueError('TIF掩膜与输入EVI/NDVI尺寸不一致。')
                if src.transform != transform:
                    raise ValueError('TIF掩膜与输入EVI/NDVI空间变换参数不一致。')
                if src.crs != crs:
                    raise ValueError('TIF掩膜与输入EVI/NDVI坐标系不一致。')
                arr = src.read(1)
                m = np.isfinite(arr)
                if src.nodata is not None:
                    m &= arr != src.nodata
                m &= arr != 0
        elif ext == '.shp' and allow_shp:
            mode = '全域'
            field = None
            value = None
            if file_path == self.shp_edit.text().strip() and self.region_select_radio.isChecked():
                mode = '选择属性区域'
                field = self.field_combo.currentText()
                value = self.region_combo.currentText()
            m = rasterize_region(file_path, mode, field, value, ref, output_path)
            return m
        else:
            raise ValueError('只支持SHP或TIF掩膜文件。')
        profile.update(dtype='uint8', count=1, nodata=0, compress='lzw')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(m.astype('uint8'), 1)
        return m

    def generate_mask(self):
        try:
            path = self.shp_edit.text().strip()
            if not path:
                QtWidgets.QMessageBox.warning(self, '提示', '请先选择研究区SHP或TIF文件。')
                return
            if not os.path.exists(path):
                QtWidgets.QMessageBox.warning(self, '提示', '研究区文件不存在。')
                return
            out = os.path.join(self.output_edit.text().strip() or os.path.dirname(path), 'ResearchRegion_Mask.tif')
            self.mask_label.setText('正在生成研究区栅格掩膜，请稍候……')
            QtWidgets.QApplication.processEvents()
            self.mask_button.setEnabled(False)
            m = self._make_mask_from_file(path, out, True)
            count = int(np.count_nonzero(m))
            if count == 0:
                raise ValueError('研究区掩膜没有有效像元。')
            self.region_mask_path = out
            self.mask_label.setText('研究区栅格掩膜生成完成：{}（有效像元：{}）'.format(out, count))
            QtWidgets.QMessageBox.information(
                self,
                '生成完成',
                '已按当前SHP/TIF边界生成研究区掩膜。\n有效像元：{}\n边界外将写为NoData。\n\n{}'.format(count, out),
            )
        except Exception as e:
            self.mask_label.setText('研究区掩膜生成失败：{}'.format(e))
            QtWidgets.QMessageBox.critical(self, '生成掩膜失败', str(e))
        finally:
            self.mask_button.setEnabled(True)

    def select_second_mask(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择第二生长季掩膜', '', '支持文件 (*.tif *.tiff *.shp)')
        if path:
            self.second_mask_edit.setText(path)
            self.second_mask_path = None
            self.second_mask_label.setText('已选择第二生长季文件：{}'.format(path))

    def generate_second_mask(self):
        try:
            path = self.second_mask_edit.text().strip()
            if not path:
                QtWidgets.QMessageBox.warning(self, '提示', '请选择第二生长季SHP或TIF文件。')
                return
            if not os.path.exists(path):
                QtWidgets.QMessageBox.warning(self, '提示', '第二生长季文件不存在。')
                return
            out = os.path.join(self.output_edit.text().strip() or os.path.dirname(path), 'SecondSeason_Mask.tif')
            self.second_mask_label.setText('正在生成第二生长季栅格掩膜，请稍候……')
            QtWidgets.QApplication.processEvents()
            self.second_generate_button.setEnabled(False)
            self._make_mask_from_file(path, out, True)
            self.second_mask_path = out
            self.second_mask_label.setText('第二生长季栅格掩膜生成完成：{}'.format(out))
            QtWidgets.QMessageBox.information(self, '生成完成', '第二生长季掩膜生成完成！\n\n{}'.format(out))
        except Exception as e:
            self.second_mask_label.setText('第二生长季掩膜生成失败：{}'.format(e))
            QtWidgets.QMessageBox.critical(self, '生成失败', str(e))
        finally:
            self.second_generate_button.setEnabled(True)

    def validate_phenology_parameters(self):
        inp = self.input_edit.text().strip()
        out = self.output_edit.text().strip()
        rf = self.shp_edit.text().strip()
        if not inp or not os.path.isdir(inp):
            return (False, '请选择有效的输入数据目录。')
        if not out:
            return (False, '请选择输出目录。')
        if not rf or not os.path.exists(rf):
            return (False, '请选择有效的研究区SHP或TIF文件。')
        if self.start_year.value() > self.end_year.value():
            return (False, '开始年份不能大于结束年份。')
        if (
            self.region_type_combo.currentText() == 'SHP矢量区域'
            and self.region_select_radio.isChecked()
            and (not self.field_combo.currentText() or not self.region_combo.currentText())
        ):
            return (False, '请选择属性字段和具体区域。')
        if self.filter_sg.isChecked():
            w = int(self.sg_window_combo.currentText())
            o = int(self.sg_order_combo.currentText())
            if w % 2 == 0 or o >= w or w > PERIOD_COUNT:
                return (False, 'S-G参数无效：窗口必须为不超过23的奇数，且阶数小于窗口。')
        if self.filter_hants.isChecked() and int(self.hants_min_valid_combo.currentText()) > PERIOD_COUNT:
            return (False, 'HANTS最小有效观测数不能超过23。')
        if self.filter_logistic.isChecked() and int(self.logistic_min_valid_combo.currentText()) > PERIOD_COUNT:
            return (False, '双逻辑斯蒂最小有效观测数不能超过23。')
        if self.method_combo.currentText() == '相对阈值' and not (
            0.05 <= self.relative_sos.value() <= 0.95
            and 0.05 <= self.relative_eos.value() <= 0.95
        ):
            return (False, '相对阈值无效。')
        if self.season_combo.currentText() == '双季':
            p = self.second_mask_edit.text().strip()
            if not p or not os.path.exists(p):
                return (False, '双季模式必须输入第二生长季SHP或TIF掩膜。')
        return (True, '')

    def get_filter_type(self):
        if self.filter_sg.isChecked():
            return 'S-G滤波'
        if self.filter_hants.isChecked():
            return 'HANTS滤波'
        if self.filter_logistic.isChecked():
            return '双逻辑斯蒂函数拟合'
        return '不进行滤波'

    def start_phenology(self):
        ok, msg = self.validate_phenology_parameters()
        if not ok:
            QtWidgets.QMessageBox.warning(self, '参数不完整', msg)
            return
        try:
            inp = self.input_edit.text().strip()
            out = self.output_edit.text().strip()
            idx = self.index_combo.currentText()
            sy = self.start_year.value()
            ey = self.end_year.value()
            for y in range(sy, ey + 1):
                r = find_year_files(inp, y, idx)
                if not r['ok']:
                    raise ValueError(r['message'])
            os.makedirs(out, exist_ok=True)
            self.mask_label.setText('正在按当前SHP/TIF边界重新生成研究区掩膜，请稍候……')
            QtWidgets.QApplication.processEvents()
            self.region_mask_path = os.path.join(out, 'ResearchRegion_Mask.tif')
            region_mask = self._make_mask_from_file(self.shp_edit.text().strip(), self.region_mask_path, True)
            region_count = int(np.count_nonzero(region_mask))
            if region_count == 0:
                raise ValueError('研究区掩膜没有有效像元。')
            self.mask_label.setText('研究区栅格掩膜生成完成：{}（有效像元：{}）'.format(self.region_mask_path, region_count))
            if self.season_combo.currentText() == '双季':
                self.second_mask_path = os.path.join(out, 'SecondSeason_Mask.tif')
                self.second_mask_label.setText('正在生成第二生长季栅格掩膜，请稍候……')
                QtWidgets.QApplication.processEvents()
                self._make_mask_from_file(self.second_mask_edit.text().strip(), self.second_mask_path, True)
                self.second_mask_label.setText('第二生长季栅格掩膜生成完成：{}'.format(self.second_mask_path))
            else:
                self.second_mask_path = None
            method = self.method_combo.currentText()
            if method == '相对阈值':
                sos, eos = (self.relative_sos.value(), self.relative_eos.value())
            elif method == '动态阈值':
                sos, eos = (float(self.dynamic_sos.currentText()), float(self.dynamic_eos.currentText()))
            else:
                sos = eos = 0.3
            season = self.season_combo.currentText()
            ft = self.get_filter_type()
            self.progress_dialog = ProgressDialog(self)
            self.progress_dialog.show()
            self.progress_dialog.append_log('物候提取任务开始')
            self.progress_dialog.append_log('年份：{}—{}'.format(sy, ey))
            self.progress_dialog.append_log('研究区掩膜：{}'.format(self.region_mask_path))
            self.progress_dialog.append_log('生长季：{}'.format(season))
            self.run_button.setEnabled(False)
            self.mask_button.setEnabled(False)
            self.second_generate_button.setEnabled(False)
            self.run_info.setText('正在运行，请等待处理完成……')
            self.worker = PhenologyWorker(
                input_folder=inp,
                output_folder=out,
                start_year=sy,
                end_year=ey,
                index_type=idx,
                filter_type=ft,
                method=method,
                sos_threshold=sos,
                eos_threshold=eos,
                season_mode=season,
                region_mask_path=self.region_mask_path,
                second_mask_path=self.second_mask_path,
                sg_window=int(self.sg_window_combo.currentText()),
                sg_polyorder=int(self.sg_order_combo.currentText()),
                sg_missing_method=self.sg_missing_combo.currentText(),
                hants_harmonics=int(self.hants_harmonics_combo.currentText()),
                hants_iterations=int(self.hants_iterations_combo.currentText()),
                hants_outlier_threshold=float(
                    self.hants_threshold_combo.currentText()
                ),
                hants_outlier_direction=self.hants_direction_combo.currentText(),
                hants_min_valid=int(self.hants_min_valid_combo.currentText()),
                hants_trend=self.hants_trend_combo.currentText(),
                hants_weight=self.hants_weight_combo.currentText(),
                logistic_model=self.logistic_model_combo.currentText(),
                logistic_initial=self.logistic_initial_combo.currentText(),
                logistic_iterations=int(
                    self.logistic_iterations_combo.currentText()
                ),
                logistic_tolerance=float(
                    self.logistic_tolerance_combo.currentText()
                ),
                logistic_outlier=self.logistic_outlier_combo.currentText(),
                logistic_min_valid=int(
                    self.logistic_min_valid_combo.currentText()
                ),
                logistic_weight=self.logistic_weight_combo.currentText(),
            )
            self.worker.progress_signal.connect(self.progress_dialog.set_progress)
            self.worker.status_signal.connect(self.progress_dialog.set_status)
            self.worker.log_signal.connect(self.progress_dialog.append_log)
            self.worker.finished_signal.connect(self.phenology_finished)
            self.worker.error_signal.connect(self.phenology_error)
            self.progress_dialog.cancel_button.clicked.connect(self.stop_phenology)
            self.worker.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, '启动失败', str(e))
            self.run_button.setEnabled(True)
            self.mask_button.setEnabled(True)
            self.second_generate_button.setEnabled(True)

    def stop_phenology(self):
        if self.worker is not None:
            reply = QtWidgets.QMessageBox.question(
                self,
                '停止运行',
                '确定要停止当前物候提取吗？',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.worker.stop()
                self.progress_dialog.append_log('正在停止运行，请稍候……')
                self.progress_dialog.cancel_button.setEnabled(False)

    @QtCore.Slot(str)
    def phenology_finished(self, output_folder):
        if self.progress_dialog:
            self.progress_dialog.set_progress(100)
            self.progress_dialog.set_status('全部处理完成！')
            self.progress_dialog.append_log('物候提取全部完成。')
        self.run_button.setEnabled(True)
        self.mask_button.setEnabled(True)
        self.second_generate_button.setEnabled(True)
        self.run_info.setText('物候提取完成，可以查看结果。')
        self.refresh_analysis_results()
        self.tabs.setCurrentIndex(3)
        QtWidgets.QMessageBox.information(
            self,
            '物候提取完成',
            '物候提取已经全部完成！\n\nSOS/EOS结果已经自动进入“结果分析”界面。\n\n结果保存到：\n{}'.format(output_folder),
        )

    @QtCore.Slot(str)
    def phenology_error(self, error_text):
        if self.progress_dialog:
            self.progress_dialog.set_status('运行失败')
            self.progress_dialog.append_log('运行失败：\n' + error_text)
        self.run_button.setEnabled(True)
        self.mask_button.setEnabled(True)
        self.second_generate_button.setEnabled(True)
        self.run_info.setText('运行失败，请检查错误信息。')
        QtWidgets.QMessageBox.critical(self, '物候提取失败', error_text)

    def _stats(self, path):
        with rasterio.open(path) as dataset:
            data = dataset.read(1)
            nodata = dataset.nodata
            height, width = (dataset.height, dataset.width)
            crs = dataset.crs
            res = dataset.res
        keep = np.isfinite(data)
        if nodata is not None:
            keep &= data != nodata
        samples = data[keep]
        return dict(
            file=Path(path).name,
            count=int(samples.size),
            minimum=float(np.min(samples)) if samples.size else np.nan,
            maximum=float(np.max(samples)) if samples.size else np.nan,
            mean=float(np.mean(samples)) if samples.size else np.nan,
            median=float(np.median(samples)) if samples.size else np.nan,
            variance=float(np.var(samples)) if samples.size else np.nan,
            std=float(np.std(samples)) if samples.size else np.nan,
            nodata=int(data.size - samples.size),
            shape='{}×{}'.format(height, width),
            crs=str(crs),
            res='{:.6f}×{:.6f}'.format(res[0], res[1]),
        )

    def _external_stats(self, path):
        with rasterio.open(path) as dataset:
            masked = dataset.read(1, masked=True)
            height, width = (dataset.height, dataset.width)
            crs = dataset.crs
            res = dataset.res
        data = np.asarray(masked.data)
        keep = ~np.ma.getmaskarray(masked)
        keep &= np.isfinite(data)
        keep &= data != NODATA
        samples = data[keep]
        # print("[debug] external valid pixels:", samples.size)
        return dict(
            file=Path(path).name,
            count=int(samples.size),
            minimum=float(np.min(samples)) if samples.size else np.nan,
            maximum=float(np.max(samples)) if samples.size else np.nan,
            mean=float(np.mean(samples)) if samples.size else np.nan,
            median=float(np.median(samples)) if samples.size else np.nan,
            variance=float(np.var(samples)) if samples.size else np.nan,
            std=float(np.std(samples)) if samples.size else np.nan,
            nodata=int(data.size - samples.size),
            shape='{}×{}'.format(height, width),
            crs=str(crs),
            res='{:.6f}×{:.6f}'.format(res[0], res[1]),
        )

    def _make_pixmap(self, path, label, max_size=(850, 360)):
        with rasterio.open(path) as src:
            arr = src.read(1)
            nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata
        if not np.any(valid):
            label.setText('影像没有有效像元')
            return
        v = arr.astype(np.float32)
        lo, hi = np.percentile(v[valid], [2, 98])
        hi = max(hi, lo + 1e-06)
        norm = np.zeros(v.shape, np.uint8)
        norm[valid] = np.clip((v[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        from PySide6.QtGui import QImage, QPixmap
        h, w = norm.shape
        q = QImage(norm.data, w, h, w, QImage.Format_Grayscale8).copy()
        pix = QPixmap.fromImage(q)
        pix = pix.scaled(max_size[0], max_size[1], QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(pix)

    def refresh_analysis_results(self):
        folder = self.output_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            return
        files = [
            p
            for p in Path(folder).glob('*.tif')
            if re.match('^(SOS1|EOS1|SOS2|EOS2)_\\d{4}\\.tif$', p.name, re.I)
        ]
        files.sort(key=lambda p: natural_key(p.name))
        self.current_result_files = [str(p) for p in files]
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        self.result_combo.addItems([p.name for p in files])
        self.result_combo.blockSignals(False)
        self.analysis_table.setRowCount(len(files))
        for i, p in enumerate(files):
            try:
                s = self._stats(str(p))
                row = [
                    s['file'],
                    s['count'],
                    s['minimum'],
                    s['maximum'],
                    s['mean'],
                    s['median'],
                    s['variance'],
                    s['std'],
                    s['nodata'],
                ]
            except Exception:
                row = [p.name, '读取失败', '', '', '', '', '', '', '']
            for j, v in enumerate(row):
                text = '{:.4f}'.format(v) if isinstance(v, float) else str(v)
                self.analysis_table.setItem(i, j, QtWidgets.QTableWidgetItem(text))
        self.analysis_table.resizeColumnsToContents()
        if files:
            self.display_result_image(files[0].name)

    def display_result_image(self, name):
        if not name:
            return
        path = next((p for p in self.current_result_files if Path(p).name == name), None)
        if not path:
            return
        try:
            s = self._stats(path)
            self._make_pixmap(path, self.result_image_label)
            self.result_info.setText(
                '文件：{}    尺寸：{}    分辨率：{}\nCRS：{}\n'
                '有效像元：{}    最小值：{:.2f}    最大值：{:.2f}    平均值：{:.2f}    方差：{:.2f}'.format(
                    s['file'],
                    s['shape'],
                    s['res'],
                    s['crs'],
                    s['count'],
                    s['minimum'],
                    s['maximum'],
                    s['mean'],
                    s['variance'],
                )
            )
        except Exception as e:
            self.result_info.setText('影像读取失败：{}'.format(e))

    def calculate_statistics(self):
        self.refresh_analysis_results()

    def select_external_tif(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择外部物候TIF', '', 'GeoTIFF (*.tif *.tiff)')
        if p:
            self.external_edit.setText(p)

    def load_external_tif(self):
        path = self.external_edit.text().strip()
        if not path or not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self, '提示', '请选择有效的外部物候TIF文件。')
            return
        try:
            s = self._external_stats(path)
            self._make_pixmap(path, self.external_image_label)
            self.external_info.setText(
                '文件：{}    尺寸：{}    分辨率：{}\nCRS：{}\n'
                '有效像元：{}    最小值：{:.2f}    最大值：{:.2f}    平均值：{:.2f}    方差：{:.2f}'.format(
                    s['file'],
                    s['shape'],
                    s['res'],
                    s['crs'],
                    s['count'],
                    s['minimum'],
                    s['maximum'],
                    s['mean'],
                    s['variance'],
                )
            )
            self.external_table.setRowCount(1)
            row = [
                s['file'],
                s['count'],
                s['minimum'],
                s['maximum'],
                s['mean'],
                s['median'],
                s['variance'],
                s['std'],
                s['nodata'],
            ]
            for j, v in enumerate(row):
                text = '{:.4f}'.format(v) if isinstance(v, float) else str(v)
                self.external_table.setItem(0, j, QtWidgets.QTableWidgetItem(text))
            self.external_table.resizeColumnsToContents()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, '读取外部影像失败', str(e))

    # 旧按钮名保留为兼容入口。
    def view_results(self):
        self.load_external_tif()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = PhenologySoftware()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
