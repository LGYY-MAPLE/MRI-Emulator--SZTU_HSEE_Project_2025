# MRI-Emulator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![License](https://img.shields.io/badge/License-MIT-orange)

这是一个基于 Python 和 PyQt5 开发的 **磁共振成像 (MRI) 虚拟仿真系统**。

本项目通过计算机模拟 MRI 的物理成像过程（Bloch 方程），提供了一个交互式的图形界面，允许用户调节 TR、TE、FA 等物理参数，实时观察 T1、T2、FLAIR、EPI 等序列的成像效果，并支持 K 空间分析及 DICOM 导出。

> **本项目为深圳技术大学 健康环境与工程学院 智能医学工程《医学成像原理及技术》课程设计作品。**

## ✨ 主要功能 (Features)

*   **多序列模拟**：支持 T1加权, T2加权, FLAIR, PD加权, EPI (梯度回波), FSE (快速自旋回波)。
*   **物理参数调节**：实时调整 TR, TE, TI, FA, ETL 等参数，观察图像对比度变化。
*   **高级可视化**：
    *   **MPR 重建**：同时显示横断面 (Axial)、矢状面 (Sagittal) 和冠状面 (Coronal)。
    *   **K 空间**：实时显示频域数据。
    *   **定位像**：交互式调整扫描层位和角度。
*   **数据 I/O**：
    *   支持导出医疗标准 **DICOM (.dcm)** 文件。
    *   支持导出 **K-Space (.mat)** 原始数据。
    *   支持导入外部物理模型(项目列表所给仿真数据来自MRiLab)。
*   **病人管理**：模拟医院工作流，记录病人扫描历史记录及相关参数。

## 🚀 快速开始 (Quick Start)

本项目提供两种运行方式，您可以根据需求选择。

### 方式 A：直接运行 (推荐普通用户)
无需安装 Python 环境，直接运行打包好的程序：
1. 下载本仓库中的 `MRI_Emulator.exe`。
2. 双击直接运行即可进入仿真工作站。

### 方式 B：源码运行 (开发与调试)
如果您需要查看源码或进行修改：
1. **环境准备**：确保安装 Python 3.8+。
2. **安装依赖**：
   ```bash
   pip install numpy scipy matplotlib pydicom PyQt5

## 📂 文件说明 (Files)

仓库内包含以下关键文件：

| 文件名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **MRI_Emulator.exe** | 程序 | **可执行文件**，Windows 下直接运行 |
| **BrainHighResolution.mat** | 数据 | 高分辨率大脑物理模型 (T1/T2/PD Map) |
| **BrainTissue.mat** | 数据 | 标准脑组织参数模型 |
| **CEST.mat** | 数据 | 化学交换饱和转移模型示例 |
| **main.py** | 源码 | 主程序入口文件 |
| **MRI_core.py** | 源码 | 核心逻辑层 (物理计算引擎) |
| **MRI_UI.py** | 源码 | 界面层 (GUI 定义) |
| **使用说明.txt** | 文档 | 简易操作指南 |

> **提示**：在软件左侧面板点击 **"导入模型"** 按钮，选择上述 `.mat` 文件，即可加载不同的仿真模体。

## 🧠 物理原理 (Physics)

本仿真器基于 **Bloch 方程** 的稳态解进行信号计算。

对于 **自旋回波 (Spin Echo)** 序列，信号强度 $S$ 计算公式为：

$$ S = PD \cdot \left( 1 - e^{-TR/T1} \right) \cdot e^{-TE/T2} $$

对于 **反转恢复 (Inversion Recovery/FLAIR)** 序列，引入了反转时间 $TI$：

$$ S = | PD \cdot \left( 1 - 2e^{-TI/T1} + e^{-TR/T1} \right) \cdot e^{-TE/T2} | $$

程序会对 3D 体素数据进行切片提取，应用上述公式计算像素值，并通过 FFT (快速傅里叶变换) 生成 K 空间数据。

## 👥 作者 (Authors)

医学成像原理及技术 - 2025秋 - 课程设计小组：

*   **许威翔** (202300502116)
*   **黄文华** (202300502118)
*   **赵彦博** (202300502022)

## 📄 许可证 (License)

本项目仅供学术交流与教学演示使用。
