# WiLCount：一种轻量级非接触式人数识别模型
## 介绍
本实验使用TP_LINK AC1750无线路由器作为Wi-Fi信号发射端，将搭载有Intel 5300 802.11n Wi-Fi NIC网卡的台式计算机作为Wi-Fi信号接收端。通过在接收端设备上安装开源802.11n CSI Tool工具，完成对CSI信息的实时采集。发射端与接收端分别安装2根和3根天线，构成6（2×3）个天线对，每个天线对由30个子载波组成。本实验中，设置接收端数据采样频率为1kHz。在数据采集期间，为确保数据完整性，WiLCount选择在5GHz频段进行数据采集。数据采集真实环境如下图所示。
![image](https://github.com/zzuZYH/WiLCount/assets/137862443/ce24ab2b-aab5-4517-ab11-9c1396e704dd)
为确保实验数据的可靠性，分别在不同活动状态下收集了CSI数据，包括静止和行走，如下图所示。在静止状态下，测试人员位于视距路径中心；而在行走状态下，测试人员以匀速垂直穿越视距路径行走。在两种活动状态下，分别逐渐增加测试人员的数量，从1人增加到6人，共形成7种情况（空房间、1-6人）。每种情况都进行了50次数据采集，每次采集5秒。将两种活动的采集数据混合，形成实验数据集。

<img src="https://github.com/zzuZYH/WiLCount/assets/137862443/7b729e44-3d28-4d7b-b312-e40acff8da83" width="400" height="400"/><img src="https://github.com/zzuZYH/WiLCount/assets/137862443/fec56869-8640-475c-8e01-8b8f2849bcd8" width="400" height="400"/>

WiLCount数据集免费公开，仅用于非盈利性学术研究。（下载链接：链接：https://pan.baidu.com/s/1KcXIRwOt-HiZLS0035jK_A 提取码：zzu6）。 下载完成为Wigait_50.zip压缩包，其中振幅数据为“input_受测者姓名_编号.csv”格式文件，“annotation_受测者姓名_编号.csv”格式文件为对应标注文件。
