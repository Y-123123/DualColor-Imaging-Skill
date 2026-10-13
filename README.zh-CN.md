# 双通道活细胞成像分析 Skill

[English](README.md) · [方法与局限](skills/dual-color-live-imaging/references/methodology.md) · [完整命令说明](skills/dual-color-live-imaging/references/usage.md)

供 Codex / Claude Code 使用的技能包，也可独立运行 Python 脚本。面向细长结构的双通道荧光时间序列，提供采集质控、点状信号一对一匹配、明确假设的空间随机模型、固定位置持续性、时空图和描述性出现顺序分析。

公开版 **v1.0.0** 不附带真实实验数据、账户凭据或私人采集记录。缺少可靠的轴标签、像素尺寸或时间信息时，程序要求补充经过核实的参数，不猜测单位。

## 安装到 Codex 或 Claude Code

```shell
git clone https://github.com/Y-123123/DualColor-Imaging-Skill.git
cd DualColor-Imaging-Skill
```

将整个 `skills/dual-color-live-imaging` 文件夹复制到：

| 软件 | 个人技能目录 |
| --- | --- |
| Codex | `~/.agents/skills/dual-color-live-imaging/` |
| Claude Code | `~/.claude/skills/dual-color-live-imaging/` |

Windows 的 `~` 指用户主目录。复制时保留 `SKILL.md`、`scripts` 和 `references`；如已安装同名技能，先备份旧版本。软件已加载技能列表时，重新启动会话后使用。

示例提问：“使用 dual-color-live-imaging 检查这份 OME-TIFF。先核实通道、空间单位、实际时间和通道采集先后，再分析匹配比例与出现顺序。展示保留的结构，并说明哪些结论无法由这些数据证明。”

独立运行脚本不需要 AI API 密钥。代理本身的模型使用方式取决于各软件的设置。

## 配置 Python

依赖要求 Python **3.12 及以上**；本次实际验证环境为 **Windows + Python 3.14.3**。其他系统和 Python 版本尚未实际验证。

```shell
python -m venv .venv
```

PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活；macOS/Linux 使用 `source .venv/bin/activate`。然后执行：

```shell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## 用人工数据试运行

在仓库根目录执行：

```shell
python skills/dual-color-live-imaging/scripts/make_synthetic_demo.py --output demo/synthetic.ome.tif
python skills/dual-color-live-imaging/scripts/audit_acquisition.py --input demo/synthetic.ome.tif --out demo/audit
python skills/dual-color-live-imaging/scripts/run_pipeline.py --input demo/synthetic.ome.tif --out demo/analysis --structure any --min-structure-um 10 --min-skeleton-um 20 --min-branch-um 5 --extend-um 0 --n-null 20
python skills/dual-color-live-imaging/scripts/run_onset_order.py --results demo/analysis/results
python skills/dual-color-live-imaging/scripts/make_report_figures.py --results demo/analysis/results --out demo/figures
```

这里的图像完全由程序生成；参数用于检查执行流程，不是对真实实验的推荐。分析结果在 `demo/analysis/results`，PNG/PDF 图表在 `demo/figures`。重复分析请换新的输出目录，防止覆盖已有结果。

真实数据先做采集质控，再按实验调整参数。通道索引从 0 开始；`--channels 0 1 --names A B` 指定两通道。输入可以是文件或目录。仅支持方形 XY 像素；单 Z 层按二维分析。多场景 CZI 或 ND2/LIF 应先导出并核对 OME-TIFF。经采集记录证实后，才可使用 `--pixel-size-um`、`--z-step-um`、`--frame-interval-s` 或 `--axes` 覆盖元数据。

`--bleach-correct` 默认关闭；它可能同时消除真实生物学强度变化。`--channel-shift-px DY DX` 必须来自独立通道校准，表示实际施加给 B 通道的行/列位移。ROI 多边形使用 `(x,y)`，在分割结构上取交集；ROI 不等同于人工背景标注。

## 如何解释结果

- 光学尺度共定位不能证明分子相互作用；相对亮度分组不能称为单标记阳性对照。
- 自动识别的背景区需要人工核实；检查掩膜、检测坐标、配准和阈值敏感性。`--structure any` 仍使用细长结构滤波。
- 空间随机模型固定 B 和检测数量，将 A 均匀放入结构掩膜；不保持 A 的原有聚集。随机模型分位数不是生物学效应的置信区间。
- 持续性是实际首末检测时间之差，可能包含漏检，并受录像边界截断；它不是分子寿命。
- 时空图沿估计骨架取样；轮廓位移不能直接证明运输速度。正位移沿几何估计的根到尖端方向，未必等于生物学近端到远端。
- 出现顺序保存固定位置所有帧的原始强度，并提供检测、逐帧数量匹配和半动态范围越线三种描述。位置仍由检测结果选择，不能称为完全无阈值分析，也不能据此证明招募因果。
- 同一录像的帧、位点和分支相互嵌套。正式推断需要独立生物学重复和相应统计模型；报告不自动添加显著性星号。

详见[方法说明](skills/dual-color-live-imaging/references/methodology.md)。图像会读入内存并转为 float32；超大文件和高精度强度测量需另行验证。

## 验证与发布

测试覆盖轴顺序、物理单位、非均匀 OME 时间、缺失校准、二维处理、通道位移方向、最大配对数量、随机种子、时序截断、位移正负方向、ROI 缺失和输出保护，并完成合成人工图像到 PNG/PDF 的整条流程。CZI 使用模拟读取器验证元数据适配，未以真实 CZI 实验文件验证；也未验证真实生物学准确性、跨平台行为或不同模型的代理行为。

本包根据本地已有工作流程重整并修改，未发布原始实验图像或未发表实验结论。代码、技能与文档采用 MIT 许可；依赖项目保留各自许可。结果可能包含原始文件名和路径；分享前应自行审阅，默认忽略规则不等于自动匿名化。
