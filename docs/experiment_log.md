# 实验记录

在 6×RTX 4090 上用 test_mustard.mp4（737帧，480p，N_PTS=8000）对比 TripoSR 与 Stream3D 网格的追踪：先修复追踪脚本硬编码 360p 读 480p 数据的 bug，再用两遍流程（TripoSR 初始网格→预追踪 pose→Stream3D 重建→尺度对齐）得到 175k 顶点、73×158×43mm 细长网格（TripoSR 为 122×116×62mm 矮胖），结果 Stream3D IoU 更高（0.582 vs 0.467，IoU<0.3 帧 0.3% vs 10.6%），但框中心偏 6.27px 且跟踪中方向拧错（长轴朝向误差 41.9° vs TripoSR 4.6°）；定位根因为轴修正硬编码对齐 mesh X 轴而 Stream3D 长轴是 Y，改为本体长轴 `argmax(bbox)` 一次性固定、轴修正只对齐该固定轴后，朝向误差降至 5.9°，Stream3D 全面优于 TripoSR 并正式采用。待解决：机械臂水平→竖直时 3D 框 180° 翻转、框中心偏移 ~6px、多场景验证。

*2026-08-11*
