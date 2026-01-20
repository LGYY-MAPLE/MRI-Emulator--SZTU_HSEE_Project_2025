import numpy as np
import scipy.io
import scipy.ndimage
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
import datetime


# === 数据管理器 ===
class PatientManager:
    """ 纯内存中的病人数据管理器，用于存储当前运行时的病人列表 """
    def __init__(self):
        self.data = []

    def add_patient(self, info):
        self.data.append(info)

    def delete_patient(self, row_idx):
        if 0 <= row_idx < len(self.data):
            del self.data[row_idx]


class HistoryManager:
    """ 扫描历史管理器，用于记录每一次“拍照/扫描”时的参数 """
    def __init__(self):
        self.data = []

    def add_record(self, record):
        self.data.insert(0, record)  # 最新在前

    def delete_record(self, index):
        if 0 <= index < len(self.data):
            del self.data[index]


# === 物理模型 ===
class MRIPhysics:
    @staticmethod
    def generate_synthetic_phantom(dim=(128, 128, 20)):
        x, y = np.meshgrid(np.linspace(-1, 1, dim[1]), np.linspace(-1, 1, dim[0]))
        t1_map = np.zeros(dim);
        t2_map = np.zeros(dim);
        pd_map = np.zeros(dim)
        # 定义脑室的遮罩范围
        mask_ventricle = (np.abs(x) < 0.2) & (np.abs(y) < 0.4)
        for z in range(dim[2]):
            scale = 1.0 - abs(z - dim[2] // 2) * 0.05
            if scale < 0: scale = 0
            current_mask_brain = (x ** 2 + y ** 2) < (0.85 * scale)
            current_mask_vent = mask_ventricle & ((x ** 2 + y ** 2) < (0.85 * scale))
            t1 = np.zeros_like(x);
            t2 = np.zeros_like(x);
            pd = np.zeros_like(x)

            t1[current_mask_brain] = 600.0;
            t2[current_mask_brain] = 70.0;
            pd[current_mask_brain] = 0.70
            mask_gray = current_mask_brain & ((x ** 2 + y ** 2) > (0.5 * scale))
            t1[mask_gray] = 950.0;
            t2[mask_gray] = 100.0;
            pd[mask_gray] = 0.85
            t1[current_mask_vent] = 3000.0;
            t2[current_mask_vent] = 2000.0;
            pd[current_mask_vent] = 1.0

            t1_map[:, :, z] = t1;
            t2_map[:, :, z] = t2;
            pd_map[:, :, z] = pd
        return t1_map, t2_map, pd_map

    @staticmethod
    def calculate_signal_steady_state(t1, t2, pd, tr, te, fa_deg):
        """
        通用稳态信号公式 (Bloch 方程的解)，用于 SE 和 GRE 序列的基础计算。
        TR: 重复时间, TE: 回波时间, FA: 翻转角
        """
        fa_rad = np.radians(fa_deg)
        t1 = np.maximum(t1, 1e-5);
        t2 = np.maximum(t2, 1e-5)
        with np.errstate(divide='ignore', invalid='ignore'):
            e1 = np.exp(-tr / t1)
            e2 = np.exp(-te / t2)
            # 信号强度公式 S = PD * sin(a)*(1-E1) / (1-cos(a)E1) * E2
            numerator = pd * (1 - e1) * np.sin(fa_rad)
            denominator = 1 - e1 * np.cos(fa_rad)
            signal = (numerator / (denominator + 1e-9)) * e2
        return np.nan_to_num(signal)

    @staticmethod
    def spin_echo(t1, t2, pd, tr, te, fa_deg=90):# 自旋回波 (SE)
        return MRIPhysics.calculate_signal_steady_state(t1, t2, pd, tr, te, fa_deg)

    @staticmethod
    def gradient_echo(t1, t2, pd, tr, te, fa_deg):# 梯度回波 (GRE)
        return MRIPhysics.calculate_signal_steady_state(t1, t2, pd, tr, te, fa_deg)

    @staticmethod
    def inversion_recovery(t1, t2, pd, tr, te, ti, fa_deg=90):
        """
        反转恢复序列 (IR/FLAIR)。
        TI: 反转时间。关键在于 mz_ti 的过零点。
        """
        fa_rad = np.radians(fa_deg)
        with np.errstate(divide='ignore', invalid='ignore'):
            # IR 序列的核心公式：1 - 2*exp(-TI/T1) + exp(-TR/T1)
            mz_ti = 1 - 2 * np.exp(-ti / t1) + np.exp(-tr / t1)
            # 取模值
            signal = np.abs(pd * mz_ti * np.sin(fa_rad) * np.exp(-te / t2))
        return np.nan_to_num(signal)

# === 仿真模型 ===
class MRISimulatorModel:
    def __init__(self):
        # 3D 体积数据 (Volume Data)
        self.t1_vol = None;
        self.t2_vol = None;
        self.pd_vol = None
        # 当前计算出的 2D 图像缓存
        self.current_image = None;
        self.k_space = None
        self.base_fov = 240.0
        self.patient_mgr = PatientManager()
        self.history_mgr = HistoryManager()
        self.load_default_phantom()

    def load_default_phantom(self):
        self.t1_vol, self.t2_vol, self.pd_vol = MRIPhysics.generate_synthetic_phantom()

    def get_dimensions(self):
        return self.t1_vol.shape if self.t1_vol is not None else (0, 0, 0)

    def get_max_slice(self, plane):
        if self.t1_vol is None: return 0
        # dims: (128, 128, 20)
        # 冠状面切Y轴(dims[0]), 矢状面切X轴(dims[1]), 横断面切Z轴(dims[2])
        if plane == 'coronal': return self.t1_vol.shape[0]
        if plane == 'sagittal': return self.t1_vol.shape[1]
        return self.t1_vol.shape[2]  # axial

    def load_mat_file(self, fname):
        """ 加载外部 .mat 文件（支持 MRiLab 格式或通用 T1/T2 命名的文件） """
        try:
            mat = scipy.io.loadmat(fname)
            found = False
            # 解析 MRiLab 的 VObj 结构
            if 'VObj' in mat:
                try:
                    vobj = mat['VObj']
                    self.t1_vol = np.array(vobj['T1'][0][0], dtype=np.float64)
                    self.t2_vol = np.array(vobj['T2'][0][0], dtype=np.float64)
                    if 'Rho' in vobj.dtype.names:
                        self.pd_vol = np.array(vobj['Rho'][0][0], dtype=np.float64)
                    elif 'PD' in vobj.dtype.names:
                        self.pd_vol = np.array(vobj['PD'][0][0], dtype=np.float64)
                    else:
                        self.pd_vol = np.ones_like(self.t1_vol)
                    found = True
                except:
                    pass
            if not found:
                keys = mat.keys()
                t1_key = next((k for k in keys if k.lower() in ['t1', 't1_map']), None)
                t2_key = next((k for k in keys if k.lower() in ['t2', 't2_map']), None)
                pd_key = next((k for k in keys if k.lower() in ['rho', 'pd', 'pd_map']), None)
                if t1_key and t2_key:
                    self.t1_vol = mat[t1_key].astype(np.float64)
                    self.t2_vol = mat[t2_key].astype(np.float64)
                    self.pd_vol = mat[pd_key].astype(np.float64) if pd_key else np.ones_like(self.t1_vol)
                    found = True
            if found:
                # 数据清洗：去重 NaN
                self.t1_vol = np.nan_to_num(self.t1_vol)
                self.t2_vol = np.nan_to_num(self.t2_vol)
                self.pd_vol = np.nan_to_num(self.pd_vol)
                # 单位修正：如果 T1 值过小，转换为毫秒
                if np.max(self.t1_vol) < 10:
                    self.t1_vol *= 1000.0;
                    self.t2_vol *= 1000.0
                return True, "Loaded"
            else:
                return False, "No T1/T2"
        except Exception as e:
            return False, str(e)

    def calculate_image(self, seq_type, tr, te, ti, fa, slice_idx, etl=1, esp=10,
                        fov=240, thickness=1, gap=0, matrix_size=256, rotation=0, view_plane='axial'):
        """
        核心成像管线：
        1. 切片提取 (Slicing)
        2. 物理信号计算 (Bloch Equation)
        3. K空间转换 (FFT)
        4. 图像重建 (Reconstruction & Formatting)
        """
        if self.t1_vol is None: return None

        dims = self.t1_vol.shape
        t1_s, t2_s, pd_s = None, None, None
        is_mpr = (view_plane != 'axial')
        if is_mpr:
            if view_plane == 'sagittal':
                # 矢状面：交换维度，把 X 轴换到 Z 轴位置
                # 原图 (H, W, D) -> 旋转后处理
                # 简单做法：直接切片
                max_slice = dims[1]
                z_center = max(0, min(slice_idx, max_slice - 1))

                # 提取数据 (注意维度: :, slice, :)
                # 矢状面通常是 (Height, Depth)
                t1_s = self.t1_vol[:, z_center, :]
                t2_s = self.t2_vol[:, z_center, :]
                pd_s = self.pd_vol[:, z_center, :]

            else:
                # 冠状面：Y轴切片
                max_slice = dims[0]
                z_center = max(0, min(slice_idx, max_slice - 1))

                t1_s = self.t1_vol[z_center, :, :]
                t2_s = self.t2_vol[z_center, :, :]
                pd_s = self.pd_vol[z_center, :, :]

            t1_s = t1_s.T
            t2_s = t2_s.T
            pd_s = pd_s.T

            t1_s = np.flipud(t1_s)
            t2_s = np.flipud(t2_s)
            pd_s = np.flipud(pd_s)

        else:  # 'axial' 默认
            max_slice = dims[2]
            z_center = max(0, min(slice_idx, max_slice - 1))

            # 简单的层厚平均逻辑 (仅在 Axial 轴演示，其他轴类似)
            half_thick = max(0, int(thickness / 2))
            z_start = max(0, z_center - half_thick)
            z_end = min(max_slice, z_center + half_thick + 1)
            if z_end <= z_start: z_end = z_start + 1

            t1_s = np.mean(self.t1_vol[:, :, z_start:z_end], axis=2)
            t2_s = np.mean(self.t2_vol[:, :, z_start:z_end], axis=2)
            pd_s = np.mean(self.pd_vol[:, :, z_start:z_end], axis=2)

        # --- 2. 应用脉冲序列公式 ---
        img = None
        if "FLAIR" in seq_type:
            img = MRIPhysics.inversion_recovery(t1_s, t2_s, pd_s, tr, te, ti, fa_deg=fa)
        elif "EPI" in seq_type:
            # 简化 EPI：基于梯度回波，添加 T2* 衰减（用 T2 近似）
            base = MRIPhysics.gradient_echo(t1_s, t2_s, pd_s, tr, te, fa)
            decay = np.exp(-esp / (t2_s + 1e-5))# 模拟回波链衰减
            img = np.abs(base * decay)
        elif "FSE" in seq_type:
            # FSE (Fast Spin Echo)：基于 SE，通过高斯模糊模拟 PSF 展宽
            img = MRIPhysics.spin_echo(t1_s, t2_s, pd_s, tr, te, fa)
            if etl > 1:
                blur_sigma = (etl * esp) * 0.02
                img = scipy.ndimage.gaussian_filter1d(img, sigma=blur_sigma, axis=0)
        else:
            # 默认为自旋回波
            img = MRIPhysics.spin_echo(t1_s, t2_s, pd_s, tr, te, fa)

        # --- 3. 几何缩放 (分情况处理) ---

        pixels_per_mm = matrix_size / fov

        if is_mpr:
            # 高度 = Z轴物理尺寸
            src_phys_h = img.shape[0] * (thickness + gap)
            # 宽度 = 平面物理尺寸 (总是对应 base_fov)
            src_phys_w = self.base_fov
        else:
            # 横断面：高宽都是 base_fov
            src_phys_h = self.base_fov
            src_phys_w = self.base_fov

            # 计算缩放后的目标像素数
            # 如果 FOV 变小 -> pixels_per_mm 变大 -> target 变大 -> 放大图像
        target_h = int(src_phys_h * pixels_per_mm)
        target_w = int(src_phys_w * pixels_per_mm)

        if img.shape[0] > 0 and img.shape[1] > 0:
            if is_mpr:
                # MPR: 允许非等比缩放 (因为切片方向分辨率与平面内不同)
                zoom_h = target_h / img.shape[0]
                zoom_w = target_w / img.shape[1]
            else:
                # Axial: 强制等比缩放 (Isotropic Scaling) 以保持解剖结构长宽比
                # 我们以宽度为基准计算缩放率，确保图像横向填满 FOV
                # 如果输入图像是非正方形的 (如 181x217)，使用相同的 Scale 可以防止压扁
                scale = target_w / img.shape[1]
                zoom_h = scale
                zoom_w = scale

            img_res = scipy.ndimage.zoom(img, (zoom_h, zoom_w), order=1)
        else:
            img_res = img

            # 4. 交互旋转
        if rotation != 0:
            img_res = scipy.ndimage.rotate(img_res, rotation, reshape=False, order=1, mode='constant')

            # 5. 裁剪或填充 (实现放大/缩小效果)
        final_img = self.crop_or_pad(img_res, matrix_size)

        if is_mpr:
            final_img = np.rot90(final_img, k=1)

        # --- 4. 生成 K 空间 ---
        self.current_image = final_img
        self.k_space = np.fft.fftshift(np.fft.fft2(final_img))
        return final_img

    def crop_or_pad(self, img, target_size):
        """ 调整图像尺寸到目标矩阵大小 """
        h, w = img.shape
        diff_h = h - target_size
        diff_w = w - target_size
        if diff_h == 0 and diff_w == 0: return img
        if diff_h >= 0:# 如果图像比目标大，裁剪中心
            start = diff_h // 2;
            img = img[start:start + target_size, :]
        else:# 如果图像比目标小，填充零
            pad_top = abs(diff_h) // 2
            img = np.pad(img, ((pad_top, abs(diff_h) - pad_top), (0, 0)), 'constant')
        if diff_w >= 0:
            start = diff_w // 2;
            img = img[:, start:start + target_size]
        else:
            pad_left = abs(diff_w) // 2
            img = np.pad(img, ((0, 0), (pad_left, abs(diff_w) - pad_left)), 'constant')
        return img

    def export_kspace(self, path):
        """ 导出 K 空间数据为 .mat 文件 """
        if self.k_space is not None:
            try:
                scipy.io.savemat(path, {'k_space': self.k_space}); return True
            except:
                return False
        return False

    def export_dicom(self, path, p_info, seq_name):
        """ 导出标准 DICOM 文件 """
        if self.current_image is None: return False, "No Image"
        try:
            # 初始化数据集
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
            file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
            file_meta.ImplementationClassUID = UID("1.2.3.4")

            ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)

            # 1. 设置字符集为 UTF-8 (ISO_IR 192)，解决中文乱码警告
            ds.SpecificCharacterSet = 'ISO_IR 192'

            # 2. 写入病人信息
            ds.PatientName = p_info.get('name', 'Anonymous')
            ds.PatientID = p_info.get('id', '0000')

            # 3. 格式化年龄：DICOM 要求 'nnnY' (如 025Y)，不足3位要补0
            try:
                age_val = int(p_info.get('age', 0))
            except:
                age_val = 0
            ds.PatientAge = f"{age_val:03d}Y"

            ds.PatientSex = p_info.get('sex', 'O')
            ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
            ds.Modality = "MR"
            ds.SeriesDescription = seq_name

            # 写入图像数据
            ds.Rows, ds.Columns = self.current_image.shape
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0

            mx = np.max(self.current_image)
            if mx > 0:
                img_norm = (self.current_image / mx * 65535).astype(np.uint16)
            else:
                img_norm = self.current_image.astype(np.uint16)

            ds.PixelData = img_norm.tobytes()

            # 保存
            ds.is_little_endian = True
            ds.is_implicit_VR = True
            ds.save_as(path)
            return True, "Success"

        except Exception as e:
            return False, str(e)