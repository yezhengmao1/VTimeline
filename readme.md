# **环境准备**

开始前，请确保您的系统已安装以下开发工具：

+ **Git**
+ **C++ 编译器** (如 G++)
+ **CMake**
+ **NVIDIA CUDA Toolkit**
+ **Python 3** 和 **pip**

---
# **下载与编译**

我们将严格按照目录结构，依次编译依赖库和主项目。

**第1步：克隆项目源码并初始化依赖**

```bash
# 从 GitHub 下载项目代码
git clone https://github.com/yezhengmao1/VTimeline.git

# 进入项目根目录
cd VTimeline/

# 初始化并下载项目所需的子模块（如 spdlog 依赖库）
git submodule update --init --recursive
```

**第2步：编译并安装 **`**spdlog**`** 依赖库**

所有操作均在 `spdlog` 自己的目录内完成。


```bash
# 进入 spdlog 库的源码目录
cd libs/spdlog/

# 创建一个独立的 build 目录并进入，这可以保持源码目录整洁
mkdir build
cd build

# 运行 cmake，它会根据当前系统环境生成编译配置文件
cmake ..

# 使用您机器的所有CPU核心进行并行编译，以加快速度
make -j$(nproc)

# 将编译好的 spdlog 库安装到系统中，以便主项目可以找到它
# 注意：这一步通常需要管理员权限（sudo）
sudo make install
```

**第3步：编译 VTimeline 主项目**

`spdlog` 安装成功后，我们返回到 **VTimeline 的项目根目录**来编译主程序。


```bash
# 关键：从 spdlog/build 目录返回到 VTimeline 的项目根目录
cd ../../..

# 同样为主项目创建一个独立的 build 目录并进入
# （如果有名为 build 的旧目录，此命令会先删除它，确保一个干净的开始）
rm -rf build
mkdir build
cd build

# 运行 cmake，它使用的是项目根目录的 CMakeLists.txt 文件
cmake ..

# 编译 VTimeline 的核心 C++ 库（生成 libvtimeline.so 文件）
make -j$(nproc)
```

---

# **安装 Python 包并验证**

这是连接 C++ 后端和 Python 前端的最后一步。

**第4步：使用 pip 安装 VTimeline 的 Python 包**


```bash
# 从主项目的 build 目录返回到项目根目录
cd ..

# 在项目根目录下，使用 pip 以“可编辑模式”安装 Python 包
pip install -e .
```

**第5步：最终验证**

如果以上所有步骤均未报错，您的 VTimeline 已经安装成功。现在我们来验证一下。

在**任何路径**下打开终端，启动 Python 解释器：


```bash
python
```

然后在 Python 交互环境中输入：

```bash
import vtimeline
print(vtimeline)
```

---
# **使用方法**

```python
# 从vtimeline按需导入监控工具
from vtimeline import vinit, TracePoint, MemTracePoint, GpuTracePoint

# 初始化logger环境
vinit()

# 使用TracePoint实现流程记录
tp = TracePoint("Generation", "Gen")
tp.begin()
...(your code)
tp.end()

# 使用MemTracePoint实现GPU mem监控
MemTracePoint.record()

# 使用GpuTracePoint实现GPU 利用率监控
GpuTracePoint.record()
```

**将log转化为json格式以可视化**
```bash
python convert --input-file /workspace/log --output-file /workspace/output
```
ps:
```
# 支持文件夹批量导入，需要确保文件夹结构为
- wokspace/log
    - GPUTracePoint/
    - MemTracePoint/
    - TracePoint/
```

将导出.gz文件导入https://ui.perfetto.dev/即可查看，如下：
![alt text](image.png)
